#include "commux/process_group_ucx.hpp"

#include <cstring>
#include <stdexcept>
#include <thread>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#ifdef COMMUX_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace commux {

ucp_tag_t make_ucp_tag(int sender_rank, uint32_t user_tag, uint32_t sub_index,
                       bool collective) {
  ucp_tag_t t = 0;
  t |= (static_cast<ucp_tag_t>(sender_rank & 0xFFFF) << 48);
  t |= (static_cast<ucp_tag_t>(sub_index & 0x7FFF) << 33);
  t |= (static_cast<ucp_tag_t>(collective ? 1u : 0u) << 32);
  t |= static_cast<ucp_tag_t>(user_tag);
  return t;
}

namespace {

// UCX memory type for a tensor's storage. Works without linking the CUDA
// runtime -- Tensor::is_cuda() only inspects the device type.
ucs_memory_type_t mem_type_of(const at::Tensor& t) {
  return t.is_cuda() ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
}

// UCX moves CUDA memory on its own stream, unordered with respect to the torch
// stream that produced/consumes the buffer. Block until the tensor's current
// stream is idle before handing the buffer to UCX, so the producer kernel has
// finished (send) / no stale kernel is mid-write (recv). nop for CPU tensors
// and for builds without CUDA.
void sync_stream_if_cuda(const at::Tensor& t) {
#ifdef COMMUX_WITH_CUDA
  if (t.is_cuda()) {
    c10::cuda::getCurrentCUDAStream(t.device().index()).synchronize();
  }
#else
  (void)t;
#endif
}

std::string addr_key(int rank) { return "commux/addr/" + std::to_string(rank); }

// Turn the pointer returned by ucp_tag_*_nbx into either:
//   nullptr  -> completed inline (nothing to wait on)
//   pointer  -> in-flight request handle
// or throw if UCX reported an immediate error.
void* normalize_request(void* req, const char* what) {
  if (req == nullptr) return nullptr;
  if (UCS_PTR_IS_ERR(req)) {
    ucs_status_t st = UCS_PTR_STATUS(req);
    TORCH_CHECK(false, "commux ", what, " failed: ", ucs_status_string(st));
  }
  return req;
}

// In-place local reduction used by the collective algorithms. Dispatches to
// CPU or CUDA automatically because it is expressed in ATen ops.
void apply_reduce(at::Tensor& acc, const at::Tensor& other, c10d::ReduceOp op) {
  switch (op.op_) {
    case c10d::ReduceOp::SUM:
      acc.add_(other);
      break;
    case c10d::ReduceOp::PRODUCT:
      acc.mul_(other);
      break;
    case c10d::ReduceOp::MIN:
      acc.copy_(at::minimum(acc, other));
      break;
    case c10d::ReduceOp::MAX:
      acc.copy_(at::maximum(acc, other));
      break;
    default:
      TORCH_CHECK(false, "commux backend: unsupported ReduceOp ",
                  static_cast<int>(op.op_));
  }
}

// c10d::Work wrapping a set of in-flight ucp requests. wait()/isCompleted()
// drive the shared worker's progress engine (the classic "wait_req" loop),
// serialized through the backend's worker mutex. Tensors are held to keep the
// underlying buffers alive until the transfer finishes.
class UCXWork : public c10d::Work {
 public:
  UCXWork(ucp_worker_h worker, std::mutex* mu, c10d::OpType opType,
          std::vector<void*> reqs, std::vector<at::Tensor> tensors,
          int source_rank = -1)
      : c10d::Work(-1, opType),
        worker_(worker),
        mu_(mu),
        reqs_(std::move(reqs)),
        tensors_(std::move(tensors)),
        source_rank_(source_rank) {}

  bool isCompleted() override {
    std::lock_guard<std::mutex> lock(*mu_);
    ucp_worker_progress(worker_);
    return reap_locked();
  }

  bool isSuccess() const override { return !err_; }

  std::exception_ptr exception() const override { return err_; }

  int sourceRank() const override { return source_rank_; }

  std::vector<at::Tensor> result() override { return tensors_; }

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    std::unique_lock<std::mutex> lock(*mu_);
    // Event-driven wait: reap finished requests, drain any further ready
    // completions, then block on the worker's wakeup fd until the next event
    // instead of busy-spinning ucp_worker_progress() and burning a CPU core.
    // worker_mu_ is held across ucp_worker_wait(): the worker is driven by a
    // single thread per rank (UCS_THREAD_MODE_SERIALIZED) and progressing it
    // here also advances every other in-flight Work, so sequential wait() calls
    // cannot deadlock.
    while (!reap_locked()) {
      if (ucp_worker_progress(worker_)) continue;  // advanced; re-check
      if (ucp_worker_wait(worker_) != UCS_OK) {
        // Wakeup unavailable in this UCX config: yield rather than hard-spin.
        lock.unlock();
        std::this_thread::yield();
        lock.lock();
      }
    }
    if (err_) {
      std::rethrow_exception(err_);
    }
    return true;
  }

  // CPU is a nop; the producing/consuming side already stream-synchronizes
  // before/after UCX touches a CUDA buffer (see sync_stream_if_cuda).
  void synchronize() override {}

 private:
  // Requires *mu_ held. Reaps finished requests WITHOUT advancing the worker;
  // callers drive ucp_worker_progress() themselves. Returns true once every
  // request has completed.
  bool reap_locked() {
    bool all_done = true;
    for (auto& r : reqs_) {
      if (r == nullptr) continue;
      ucs_status_t st = ucp_request_check_status(r);
      if (st == UCS_INPROGRESS) {
        all_done = false;
      } else {
        if (st != UCS_OK && !err_) {
          err_ = std::make_exception_ptr(std::runtime_error(
              std::string("commux request failed: ") + ucs_status_string(st)));
        }
        ucp_request_free(r);
        r = nullptr;
      }
    }
    return all_done;
  }

  ucp_worker_h worker_;
  std::mutex* mu_;
  std::vector<void*> reqs_;
  std::vector<at::Tensor> tensors_;
  int source_rank_;
  std::exception_ptr err_;
};

c10::intrusive_ptr<c10d::Work> completed_work(c10d::OpType opType,
                                              std::vector<at::Tensor> tensors,
                                              ucp_worker_h worker,
                                              std::mutex* mu) {
  return c10::make_intrusive<UCXWork>(worker, mu, opType, std::vector<void*>{},
                                      std::move(tensors));
}

}  // namespace

ProcessGroupUCX::ProcessGroupUCX(c10::intrusive_ptr<c10d::Store> store, int rank,
                                 int size)
    : c10d::Backend(rank, size),
      store_(std::move(store)),
      rank_(rank),
      size_(size) {
  init_ucx();
  bootstrap_endpoints();
}

ProcessGroupUCX::~ProcessGroupUCX() {
  std::lock_guard<std::mutex> lock(worker_mu_);
  for (auto ep : eps_) {
    if (ep == nullptr) continue;
    ucp_request_param_t p;
    std::memset(&p, 0, sizeof(p));
    void* req = ucp_ep_close_nbx(ep, &p);
    if (UCS_PTR_IS_PTR(req)) {
      ucs_status_t st;
      while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS) {
        if (ucp_worker_progress(worker_)) continue;
        if (ucp_worker_wait(worker_) != UCS_OK) std::this_thread::yield();
      }
      ucp_request_free(req);
    }
  }
  if (worker_ != nullptr) ucp_worker_destroy(worker_);
  if (context_ != nullptr) ucp_cleanup(context_);
}

void ProcessGroupUCX::init_ucx() {
  ucp_config_t* config = nullptr;
  ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
  TORCH_CHECK(st == UCS_OK, "ucp_config_read: ", ucs_status_string(st));

  ucp_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  // TAG: tag-matching p2p. WAKEUP: lets the wait paths sleep on the worker's
  // event fd (ucp_worker_wait) instead of busy-spinning ucp_worker_progress().
  // Every transport commux uses between distinct ranks (cuda_ipc, cuda_copy,
  // sm, tcp, ib/rc) advertises UCT_IFACE_FLAG_EVENT_FD, so wakeups arrive.
  params.features = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;

  st = ucp_init(&params, config, &context_);
  ucp_config_release(config);
  TORCH_CHECK(st == UCS_OK, "ucp_init: ", ucs_status_string(st));

  ucp_worker_params_t wp;
  std::memset(&wp, 0, sizeof(wp));
  wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wp.thread_mode = UCS_THREAD_MODE_SERIALIZED;

  st = ucp_worker_create(context_, &wp, &worker_);
  TORCH_CHECK(st == UCS_OK, "ucp_worker_create: ", ucs_status_string(st));
}

void ProcessGroupUCX::bootstrap_endpoints() {
  // Publish this worker's address through the c10d store (out-of-band
  // rendezvous). Any c10d::Store works -- a TCPStore for standalone runs, or the
  // PrefixStore torch.distributed hands to a backend.
  ucp_address_t* addr = nullptr;
  size_t addr_len = 0;
  ucs_status_t st = ucp_worker_get_address(worker_, &addr, &addr_len);
  TORCH_CHECK(st == UCS_OK, "ucp_worker_get_address: ", ucs_status_string(st));

  std::vector<uint8_t> blob(reinterpret_cast<uint8_t*>(addr),
                            reinterpret_cast<uint8_t*>(addr) + addr_len);
  store_->set(addr_key(rank_), blob);
  ucp_worker_release_address(worker_, addr);

  eps_.assign(size_, nullptr);
  for (int r = 0; r < size_; ++r) {
    if (r == rank_) continue;
    store_->wait({addr_key(r)});
    std::vector<uint8_t> remote = store_->get(addr_key(r));

    ucp_ep_params_t ep_params;
    std::memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(remote.data());

    st = ucp_ep_create(worker_, &ep_params, &eps_[r]);
    TORCH_CHECK(st == UCS_OK, "ucp_ep_create to rank ", r, ": ",
                ucs_status_string(st));
  }
}

void* ProcessGroupUCX::post_send(const at::Tensor& t, int dst, uint32_t user_tag,
                                 uint32_t sub_index, bool collective) {
  TORCH_CHECK(dst >= 0 && dst < size_ && dst != rank_,
              "commux send: invalid dst rank ", dst);
  sync_stream_if_cuda(t);
  ucp_tag_t tag = make_ucp_tag(rank_, user_tag, sub_index, collective);

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));
  p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  p.datatype = ucp_dt_make_contig(1);
  p.memory_type = mem_type_of(t);

  size_t nbytes = static_cast<size_t>(t.numel()) * t.element_size();
  void* req = ucp_tag_send_nbx(eps_[dst], t.data_ptr(), nbytes, tag, &p);
  return normalize_request(req, "tag_send");
}

void* ProcessGroupUCX::post_recv(at::Tensor& t, int src, uint32_t user_tag,
                                 uint32_t sub_index, bool collective) {
  TORCH_CHECK(t.is_contiguous(), "commux recv: destination tensor must be "
                                 "contiguous");
  sync_stream_if_cuda(t);
  ucp_tag_t tag = make_ucp_tag(src, user_tag, sub_index, collective);
  ucp_tag_t mask = ~static_cast<ucp_tag_t>(0);  // exact (sender, tag) match

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));
  p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  p.datatype = ucp_dt_make_contig(1);
  p.memory_type = mem_type_of(t);

  size_t nbytes = static_cast<size_t>(t.numel()) * t.element_size();
  void* req = ucp_tag_recv_nbx(worker_, t.data_ptr(), nbytes, tag, mask, &p);
  return normalize_request(req, "tag_recv");
}

void ProcessGroupUCX::wait_request(void* req) {
  if (req == nullptr) return;
  // Drive progress while events are ready; otherwise sleep on the worker's
  // wakeup fd instead of busy-spinning. Caller holds worker_mu_.
  ucs_status_t st;
  while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS) {
    if (ucp_worker_progress(worker_)) continue;
    if (ucp_worker_wait(worker_) != UCS_OK) std::this_thread::yield();
  }
  ucp_request_free(req);
  TORCH_CHECK(st == UCS_OK, "commux request failed: ", ucs_status_string(st));
}

void ProcessGroupUCX::coll_send(const at::Tensor& t, int dst,
                                uint32_t user_tag) {
  wait_request(post_send(t, dst, user_tag, /*sub=*/0, /*collective=*/true));
}

void ProcessGroupUCX::coll_recv(at::Tensor& t, int src, uint32_t user_tag) {
  wait_request(post_recv(t, src, user_tag, /*sub=*/0, /*collective=*/true));
}

// -------------------------------------------------------------------------
// Point-to-point
// -------------------------------------------------------------------------

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  std::vector<void*> reqs;
  std::vector<at::Tensor> keep;  // hold contiguous copies until completion
  reqs.reserve(tensors.size());
  keep.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::Tensor c = tensors[i].contiguous();
    keep.push_back(c);
    reqs.push_back(post_send(c, dstRank, static_cast<uint32_t>(tag),
                             static_cast<uint32_t>(i), /*collective=*/false));
  }
  return c10::make_intrusive<UCXWork>(worker_, &worker_mu_, c10d::OpType::SEND,
                                      std::move(reqs), std::move(keep));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  std::vector<void*> reqs;
  reqs.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    reqs.push_back(post_recv(tensors[i], srcRank, static_cast<uint32_t>(tag),
                             static_cast<uint32_t>(i), /*collective=*/false));
  }
  return c10::make_intrusive<UCXWork>(worker_, &worker_mu_, c10d::OpType::RECV,
                                      std::move(reqs), tensors);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  // Probe-then-receive so we can report the true source rank. Blocking, but
  // recvAnysource is off the halo-exchange critical path.
  std::lock_guard<std::mutex> lock(worker_mu_);
  int source_rank = -1;
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::Tensor& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "commux recvAnysource: tensor must be "
                                   "contiguous");
    sync_stream_if_cuda(t);
    ucp_tag_t want = make_ucp_tag(0, static_cast<uint32_t>(tag),
                                  static_cast<uint32_t>(i), /*collective=*/false);
    ucp_tag_recv_info_t info;
    ucp_tag_message_h msg = nullptr;
    while ((msg = ucp_tag_probe_nb(worker_, want, kTagMaskAnySource,
                                   /*remove=*/1, &info)) == nullptr) {
      if (ucp_worker_progress(worker_)) continue;
      if (ucp_worker_wait(worker_) != UCS_OK) std::this_thread::yield();
    }
    source_rank = static_cast<int>(info.sender_tag >> 48);

    ucp_request_param_t p;
    std::memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    p.datatype = ucp_dt_make_contig(1);
    p.memory_type = mem_type_of(t);
    size_t nbytes = static_cast<size_t>(t.numel()) * t.element_size();
    void* req = ucp_tag_msg_recv_nbx(worker_, t.data_ptr(), nbytes, msg, &p);
    wait_request(normalize_request(req, "tag_msg_recv"));
  }
  return c10::make_intrusive<UCXWork>(worker_, &worker_mu_,
                                      c10d::OpType::RECVANYSOURCE,
                                      std::vector<void*>{}, tensors, source_rank);
}

// -------------------------------------------------------------------------
// Collectives (linear reduce-to-root + broadcast over tagged p2p).
// Correctness-first; recursive-doubling is a future optimization. Typical
// payloads (timestep / diagnostics) are tiny, so linear is fine.
// -------------------------------------------------------------------------

void ProcessGroupUCX::reduce_locked(std::vector<at::Tensor>& tensors,
                                    c10d::ReduceOp op, int root) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::Tensor& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "commux reduce: tensor must be contiguous");
    if (rank_ == root) {
      at::Tensor tmp = at::empty_like(t);
      for (int r = 0; r < size_; ++r) {
        if (r == root) continue;
        coll_recv(tmp, r, static_cast<uint32_t>(i));
        apply_reduce(t, tmp, op);
      }
    } else {
      coll_send(t, root, static_cast<uint32_t>(i));
    }
  }
}

void ProcessGroupUCX::broadcast_locked(std::vector<at::Tensor>& tensors,
                                       int root) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::Tensor& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "commux broadcast: tensor must be contiguous");
    if (rank_ == root) {
      for (int r = 0; r < size_; ++r) {
        if (r == root) continue;
        coll_send(t, r, static_cast<uint32_t>(i));
      }
    } else {
      coll_recv(t, root, static_cast<uint32_t>(i));
    }
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (size_ > 1) {
    reduce_locked(tensors, opts.reduceOp, /*root=*/0);
    broadcast_locked(tensors, /*root=*/0);
  }
  return completed_work(c10d::OpType::ALLREDUCE, tensors, worker_, &worker_mu_);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (size_ > 1) {
    reduce_locked(tensors, opts.reduceOp, static_cast<int>(opts.rootRank));
  }
  return completed_work(c10d::OpType::REDUCE, tensors, worker_, &worker_mu_);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (size_ > 1) {
    broadcast_locked(tensors, static_cast<int>(opts.rootRank));
  }
  return completed_work(c10d::OpType::BROADCAST, tensors, worker_, &worker_mu_);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::barrier(
    const c10d::BarrierOptions& /*opts*/) {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (size_ > 1) {
    std::vector<at::Tensor> token{
        at::zeros({1}, at::TensorOptions().dtype(at::kInt))};
    reduce_locked(token, c10d::ReduceOp(c10d::ReduceOp::SUM), /*root=*/0);
    broadcast_locked(token, /*root=*/0);
  }
  return completed_work(c10d::OpType::BARRIER, {}, worker_, &worker_mu_);
}

}  // namespace commux
