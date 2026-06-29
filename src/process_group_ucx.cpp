#include "commux/process_group_ucx.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <poll.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

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

// Max idle sleep (ms) inside a blocking wait before it re-progresses the
// worker. Overridable via COMMUX_WAIT_POLL_MS for tuning latency vs. wakeup
// rate.
int wait_poll_ms() {
  static const int ms = [] {
    const char* e = std::getenv("COMMUX_WAIT_POLL_MS");
    int v = e ? std::atoi(e) : 1;
    return v > 0 ? v : 1;
  }();
  return ms;
}

// Sleep until the worker's wakeup fd signals an event OR wait_poll_ms()
// elapses, then return so the caller re-progresses. The timeout is the fix for
// a LOST-WAKEUP deadlock: when several threads each block in the worker's
// wakeup fd at once (e.g. a multi-threaded consumer like snapy's
// BlockWorkerPool driving the single shared worker), one arriving event wakes
// only ONE thread. That thread's ucp_worker_progress() completes the requests
// of the OTHERS too, but they stay asleep on the now-consumed fd with no
// further event to wake them -- so they never re-check reap()/the request
// status and hang forever. (The same starvation also strands a peer when one
// rank is in a collective and another is still in p2p.) An infinite
// ucp_worker_wait() relies entirely on that one-shot event; bounding the sleep
// means every parked thread re-progresses and observes its own completion
// regardless of who consumed the event. poll() still returns immediately when
// an event arrives, so latency is unchanged in the common case; the timeout
// only caps idle sleep, keeping CPU near zero rather than busy-spinning
// ucp_worker_progress().
void worker_wait_timed(ucp_worker_h worker) {
  ucs_status_t s = ucp_worker_arm(worker);
  if (s == UCS_ERR_BUSY) return;  // events arrived during arm -> progress now
  if (s != UCS_OK) {
    std::this_thread::yield();  // wakeup unsupported: fall back to a yield
    return;
  }
  int efd;
  if (ucp_worker_get_efd(worker, &efd) != UCS_OK) {
    std::this_thread::yield();
    return;
  }
  struct pollfd pfd;
  pfd.fd = efd;
  pfd.events = POLLIN;
  poll(&pfd, 1, wait_poll_ms());
}

// CUDA stream-sync is provided by the optional sibling libcommux_cuda.so -- the
// only commux object that links libc10_cuda.so. We dlopen it lazily so that
// libcommux.so itself has NO link-time dependency on libc10_cuda.so and thus
// loads on a CPU-only torch (where that library does not exist). On a CUDA
// torch the sibling and its libc10_cuda dependency are present, so the sync
// runs.
using commux_cuda_sync_fn = void (*)(int);

commux_cuda_sync_fn resolve_cuda_sync() {
  // Resolved once; nullptr means the shim is unavailable (CPU-only torch).
  static commux_cuda_sync_fn fn = []() -> commux_cuda_sync_fn {
    // Load the sibling sitting next to this very library (located via dladdr),
    // so it resolves regardless of LD_LIBRARY_PATH or the consumer's rpath. Its
    // own libc10_cuda.so dependency is satisfied from the already-loaded torch.
    std::string path = "libcommux_cuda.so";
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&resolve_cuda_sync), &info) &&
        info.dli_fname != nullptr) {
      std::string self(info.dli_fname);
      auto slash = self.find_last_of('/');
      if (slash != std::string::npos)
        path = self.substr(0, slash + 1) + "libcommux_cuda.so";
    }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr)  // fall back to the bare soname (LD_LIBRARY_PATH search)
      h = dlopen("libcommux_cuda.so", RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) return nullptr;
    return reinterpret_cast<commux_cuda_sync_fn>(
        dlsym(h, "commux_cuda_sync_stream"));
  }();
  return fn;
}

// UCX moves CUDA memory on its own stream, unordered with respect to the torch
// stream that produced/consumes the buffer. Block until the tensor's current
// stream is idle before handing the buffer to UCX, so the producer kernel has
// finished (send) / no stale kernel is mid-write (recv). nop for CPU tensors. A
// CUDA tensor with no sync shim available is a hard error (skipping the sync
// would be a silent data race), not a no-op.
void sync_stream_if_cuda(const at::Tensor& t) {
  if (!t.is_cuda()) return;
  commux_cuda_sync_fn fn = resolve_cuda_sync();
  TORCH_CHECK(
      fn != nullptr,
      "commux: received a CUDA tensor but the CUDA stream-sync helper "
      "(libcommux_cuda.so) could not be loaded -- this commux was built "
      "without CUDA support, or libc10_cuda.so is unavailable.");
  fn(static_cast<int>(t.device().index()));
}

// Halo exchange hands commux a whole vector of per-variable ghost tensors in
// one send()/recv() call. With COMMUX_COALESCE=1 they are merged into a single
// UCX tagged message via the IOV datatype (see send/recv).
//
// OPT-IN (default off) on purpose: coalescing is NOT a universal win. UCX's IOV
// path for CUDA is not zero-copy -- it gathers the V device buffers into a
// staging buffer -- so coalescing trades V rendezvous handshakes for
// 1 handshake + 1 gather/scatter copy + the loss of pipelining that V
// concurrent in-flight messages enjoy. Measured intra-node (cuda_ipc): faster
// for small V (V=2: 81 vs 116 us round-trip), SLOWER for large V (V=16: 481 vs
// 315 us); neutral for the bandwidth-bound CRM halo. It is most likely to pay
// off on high-latency links (InfiniBand multi-node) where handshakes dominate
// -- so it is exposed as a flag to benchmark per deployment rather than forced
// on.
bool coalesce_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("COMMUX_COALESCE");
    return e != nullptr && (std::string(e) == "1" || std::string(e) == "on");
  }();
  return v;
}

// Opt-in group window (COMMUX_GROUP=1): batch all send/recv posts of one
// exchange into a single flush (one stream-sync) at endCoalescing(), the c10d
// analog of ncclGroupStart/End. Default off -> startCoalescing() is a no-op and
// posts happen immediately (behavior-preserving).
bool group_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("COMMUX_GROUP");
    return e != nullptr && (std::string(e) == "1" || std::string(e) == "on");
  }();
  return v;
}

// True if every tensor lives in the same UCX memory space (all CUDA or all
// host); sets `mt` accordingly. IOV carries a single memory_type for the whole
// scatter-gather list, so coalescing requires homogeneity. Halo buffers are all
// on one device, so this holds; mixed batches fall back to per-tensor messages.
bool homogeneous_memtype(const std::vector<at::Tensor>& ts,
                         ucs_memory_type_t& mt) {
  if (ts.empty()) return false;
  bool cuda = ts.front().is_cuda();
  for (const auto& t : ts)
    if (t.is_cuda() != cuda) return false;
  mt = cuda ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
  return true;
}

// Synchronize the current CUDA stream of each distinct device in the batch once
// (typically a single device for a halo exchange), so the producer/consumer
// kernels are done before UCX -- which moves CUDA memory on its own stream --
// touches the buffers. Replaces the previous per-tensor sync. nop for CPU.
void sync_batch_streams(const std::vector<at::Tensor>& ts) {
  for (size_t i = 0; i < ts.size(); ++i) {
    if (!ts[i].is_cuda()) continue;
    auto dev = ts[i].device().index();
    bool seen = false;
    for (size_t j = 0; j < i; ++j)
      if (ts[j].is_cuda() && ts[j].device().index() == dev) {
        seen = true;
        break;
      }
    if (!seen) sync_stream_if_cuda(ts[i]);  // first tensor on this device
  }
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
// drive the shared worker's progress engine (the classic "wait_req" loop). The
// worker is UCS_THREAD_MODE_MULTI (internally thread-safe), so waiting on one
// Work does not hold the backend's worker mutex and does not block other local
// threads from posting/progressing more nonblocking sends/recvs. Tensors are
// held to keep the underlying buffers alive until the transfer finishes.
class UCXWork : public c10d::Work {
 public:
  UCXWork(ucp_worker_h worker, std::mutex* mu, c10d::OpType opType,
          std::vector<void*> reqs, std::vector<at::Tensor> tensors,
          int source_rank = -1, std::vector<ucp_dt_iov_t> iov = {})
      : c10d::Work(-1, opType),
        worker_(worker),
        mu_(mu),
        reqs_(std::move(reqs)),
        tensors_(std::move(tensors)),
        source_rank_(source_rank),
        iov_(std::move(iov)) {}

  bool isCompleted() override {
    // The worker is thread-safe (MULTI), so progress + reap need no lock; each
    // Work is reaped by the single thread that created it.
    ucp_worker_progress(worker_);
    return reap();
  }

  bool isSuccess() const override { return !err_; }

  std::exception_ptr exception() const override { return err_; }

  int sourceRank() const override { return source_rank_; }

  std::vector<at::Tensor> result() override { return tensors_; }

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    // Event-driven wait: reap finished requests, drain ready completions, then
    // block on the worker's wakeup fd until the next event instead of busy-
    // spinning ucp_worker_progress() and burning a CPU core. The worker is
    // UCS_THREAD_MODE_MULTI (internally thread-safe), so this path
    // intentionally does NOT hold ProcessGroupUCX::worker_mu_: a thread parked
    // here must not block other threads from posting and progressing the shared
    // worker -- their progress is precisely what completes this Work. (Holding
    // the mutex across a blocking ucp_worker_wait() would serialize, and can
    // deadlock, multi-threaded consumers such as snapy.)
    while (!reap()) {
      if (ucp_worker_progress(worker_)) continue;  // advanced; re-check
      // Idle: sleep on the worker's wakeup fd, but only until the next event or
      // a short timeout -- a bounded sleep so concurrently parked threads can't
      // miss a lost wakeup on the shared worker (see worker_wait_timed).
      worker_wait_timed(worker_);
    }
    if (err_) {
      std::rethrow_exception(err_);
    }
    return true;
  }

  // CPU is a nop; the producing/consuming side already stream-synchronizes
  // before/after UCX touches a CUDA buffer (see sync_stream_if_cuda).
  void synchronize() override {}

  // Fill a pending (empty) Work created at startCoalescing() with the requests
  // and buffers produced by the endCoalescing() flush. Caller holds *mu_.
  void adopt(std::vector<void*> reqs, std::vector<at::Tensor> tensors) {
    reqs_ = std::move(reqs);
    tensors_ = std::move(tensors);
  }

 private:
  // Reaps finished requests WITHOUT advancing the worker; callers drive
  // ucp_worker_progress() themselves. Returns true once every request has
  // completed. Safe to call lock-free: the worker is MULTI and each Work is
  // reaped by a single thread.
  bool reap() {
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
  // IOV descriptor array backing a coalesced send/recv; UCX references it until
  // the request completes, so the Work must outlive the transfer. (std::vector
  // move preserves the heap buffer, so the pointer handed to UCX stays valid.)
  std::vector<ucp_dt_iov_t> iov_;
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

ProcessGroupUCX::ProcessGroupUCX(c10::intrusive_ptr<c10d::Store> store,
                                 int rank, int size)
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
  params.field_mask =
      UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
  // TAG: tag-matching p2p. WAKEUP: lets the wait paths sleep on the worker's
  // event fd (ucp_worker_wait) instead of busy-spinning ucp_worker_progress().
  // Every transport commux uses between distinct ranks (cuda_ipc, cuda_copy,
  // sm, tcp, ib/rc) advertises UCT_IFACE_FLAG_EVENT_FD, so wakeups arrive.
  params.features = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
  // commux uses a single worker per rank, shared by multiple threads, so the
  // context itself does not strictly need cross-worker thread safety (UCX
  // docs). Set it anyway as belt-and-suspenders / future-proofing; the
  // per-worker UCS_THREAD_MODE_MULTI below is what makes the shared worker
  // thread-safe.
  params.mt_workers_shared = 1;

  st = ucp_init(&params, config, &context_);
  ucp_config_release(config);
  TORCH_CHECK(st == UCS_OK, "ucp_init: ", ucs_status_string(st));

  ucp_worker_params_t wp;
  std::memset(&wp, 0, sizeof(wp));
  wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  // MULTI: the worker is shared across threads by multi-threaded consumers
  // (e.g. snapy drives per-block exchanges on separate threads). UCX serializes
  // worker access internally, so wait() can block on it without holding
  // worker_mu_ -- letting other threads keep posting/progressing meanwhile.
  wp.thread_mode = UCS_THREAD_MODE_MULTI;

  st = ucp_worker_create(context_, &wp, &worker_);
  TORCH_CHECK(st == UCS_OK, "ucp_worker_create: ", ucs_status_string(st));

  // Verify we actually got a multi-thread-safe worker; the lock-free wait above
  // relies on it. (Bundled UCX is built --enable-mt.)
  ucp_worker_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
  st = ucp_worker_query(worker_, &attr);
  TORCH_CHECK(st == UCS_OK, "ucp_worker_query: ", ucs_status_string(st));
  TORCH_CHECK(attr.thread_mode == UCS_THREAD_MODE_MULTI,
              "commux needs a multi-thread-safe UCX worker (got thread_mode=",
              static_cast<int>(attr.thread_mode),
              "); build UCX with --enable-mt.");
}

void ProcessGroupUCX::bootstrap_endpoints() {
  // Publish this worker's address through the c10d store (out-of-band
  // rendezvous). Any c10d::Store works -- a TCPStore for standalone runs, or
  // the PrefixStore torch.distributed hands to a backend.
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

void* ProcessGroupUCX::post_send(const at::Tensor& t, int dst,
                                 uint32_t user_tag, uint32_t sub_index,
                                 bool collective) {
  TORCH_CHECK(dst >= 0 && dst < size_ && dst != rank_,
              "commux send: invalid dst rank ", dst);
  // Stream sync is the caller's responsibility (send()/coll_send() sync the
  // batch / tensor once) so a multi-tensor post does not sync per tensor.
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
  TORCH_CHECK(t.is_contiguous(),
              "commux recv: destination tensor must be "
              "contiguous");
  // Stream sync is the caller's responsibility (recv()/coll_recv() sync once).
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
  // wakeup fd (bounded, to avoid the lost-wakeup deadlock -- see
  // worker_wait_timed) instead of busy-spinning. Caller holds worker_mu_.
  ucs_status_t st;
  while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS) {
    if (ucp_worker_progress(worker_)) continue;
    worker_wait_timed(worker_);
  }
  ucp_request_free(req);
  TORCH_CHECK(st == UCS_OK, "commux request failed: ", ucs_status_string(st));
}

void ProcessGroupUCX::coll_send(const at::Tensor& t, int dst,
                                uint32_t user_tag) {
  sync_stream_if_cuda(t);
  wait_request(post_send(t, dst, user_tag, /*sub=*/0, /*collective=*/true));
}

void ProcessGroupUCX::coll_recv(at::Tensor& t, int src, uint32_t user_tag) {
  sync_stream_if_cuda(t);
  wait_request(post_recv(t, src, user_tag, /*sub=*/0, /*collective=*/true));
}

// -------------------------------------------------------------------------
// Point-to-point
// -------------------------------------------------------------------------

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  TORCH_CHECK(dstRank >= 0 && dstRank < size_ && dstRank != rank_,
              "commux send: invalid dst rank ", dstRank);
  std::lock_guard<std::mutex> lock(worker_mu_);

  if (coalescing_) {
    // Inside a group window: defer the post; flushed together at endCoalescing.
    DeferredOp op;
    op.is_send = true;
    op.peer = dstRank;
    op.tag = static_cast<uint32_t>(tag);
    op.keep.reserve(tensors.size());
    for (auto& t : tensors) op.keep.push_back(t.contiguous());
    deferred_.push_back(std::move(op));
    return pending_;
  }

  std::vector<at::Tensor> keep;  // hold contiguous copies until completion
  keep.reserve(tensors.size());
  for (auto& t : tensors) keep.push_back(t.contiguous());

  // One stream sync for the whole batch instead of one per tensor.
  sync_batch_streams(keep);

  std::vector<void*> reqs;
  ucs_memory_type_t mt;
  // Coalesce the per-neighbor vector into ONE tagged UCX message via the IOV
  // datatype (zero-copy scatter-gather): one rendezvous handshake / tag-match /
  // in-flight request instead of V. The matching recv() coalesces too -- the
  // predicate (size>1 + homogeneous memory type) depends only on tensor
  // metadata that is identical on both ranks by symmetric buffer construction,
  // so sender and receiver always agree on the single-tag (sub_index=0) layout.
  if (keep.size() > 1 && coalesce_enabled() && homogeneous_memtype(keep, mt)) {
    std::vector<ucp_dt_iov_t> iov;
    iov.reserve(keep.size());
    for (auto& c : keep)
      iov.push_back(
          {c.data_ptr(), static_cast<size_t>(c.numel()) * c.element_size()});

    ucp_request_param_t p;
    std::memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    p.datatype = ucp_dt_make_iov();
    p.memory_type = mt;
    ucp_tag_t tagv = make_ucp_tag(rank_, static_cast<uint32_t>(tag),
                                  /*sub_index=*/0, /*collective=*/false);
    void* req =
        ucp_tag_send_nbx(eps_[dstRank], iov.data(), iov.size(), tagv, &p);
    reqs.push_back(normalize_request(req, "tag_send(iov)"));
    return c10::make_intrusive<UCXWork>(worker_, &worker_mu_,
                                        c10d::OpType::SEND, std::move(reqs),
                                        std::move(keep),
                                        /*source_rank=*/-1, std::move(iov));
  }

  // Fallback: one message per tensor (single tensor, mixed memory type, or
  // COMMUX_COALESCE=0). post_send no longer syncs -- the batch sync above
  // covers it.
  reqs.reserve(keep.size());
  for (size_t i = 0; i < keep.size(); ++i)
    reqs.push_back(post_send(keep[i], dstRank, static_cast<uint32_t>(tag),
                             static_cast<uint32_t>(i), /*collective=*/false));
  return c10::make_intrusive<UCXWork>(worker_, &worker_mu_, c10d::OpType::SEND,
                                      std::move(reqs), std::move(keep));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  std::lock_guard<std::mutex> lock(worker_mu_);

  if (coalescing_) {
    DeferredOp op;
    op.is_send = false;
    op.peer = srcRank;
    op.tag = static_cast<uint32_t>(tag);
    op.keep.reserve(tensors.size());
    for (auto& t : tensors) {
      TORCH_CHECK(t.is_contiguous(),
                  "commux recv: destination tensor must be contiguous");
      op.keep.push_back(t);
    }
    deferred_.push_back(std::move(op));
    return pending_;
  }

  // One stream sync for the whole batch instead of one per tensor.
  sync_batch_streams(tensors);

  std::vector<void*> reqs;
  ucs_memory_type_t mt;
  // Symmetric to send(): scatter one coalesced message into the V recv buffers
  // via IOV. Total bytes equal the sender's by symmetric buffer construction.
  if (tensors.size() > 1 && coalesce_enabled() &&
      homogeneous_memtype(tensors, mt)) {
    std::vector<ucp_dt_iov_t> iov;
    iov.reserve(tensors.size());
    for (auto& t : tensors) {
      TORCH_CHECK(t.is_contiguous(),
                  "commux recv: destination tensor must be contiguous");
      iov.push_back(
          {t.data_ptr(), static_cast<size_t>(t.numel()) * t.element_size()});
    }

    ucp_request_param_t p;
    std::memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    p.datatype = ucp_dt_make_iov();
    p.memory_type = mt;
    ucp_tag_t tagv = make_ucp_tag(srcRank, static_cast<uint32_t>(tag),
                                  /*sub_index=*/0, /*collective=*/false);
    ucp_tag_t mask = ~static_cast<ucp_tag_t>(0);  // exact (sender, tag) match
    void* req =
        ucp_tag_recv_nbx(worker_, iov.data(), iov.size(), tagv, mask, &p);
    reqs.push_back(normalize_request(req, "tag_recv(iov)"));
    return c10::make_intrusive<UCXWork>(
        worker_, &worker_mu_, c10d::OpType::RECV, std::move(reqs), tensors,
        /*source_rank=*/-1, std::move(iov));
  }

  reqs.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i)
    reqs.push_back(post_recv(tensors[i], srcRank, static_cast<uint32_t>(tag),
                             static_cast<uint32_t>(i), /*collective=*/false));
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
    TORCH_CHECK(t.is_contiguous(),
                "commux recvAnysource: tensor must be "
                "contiguous");
    sync_stream_if_cuda(t);
    ucp_tag_t want =
        make_ucp_tag(0, static_cast<uint32_t>(tag), static_cast<uint32_t>(i),
                     /*collective=*/false);
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
  return c10::make_intrusive<UCXWork>(
      worker_, &worker_mu_, c10d::OpType::RECVANYSOURCE, std::vector<void*>{},
      tensors, source_rank);
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
    TORCH_CHECK(t.is_contiguous(),
                "commux broadcast: tensor must be contiguous");
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

// ---------------------------------------------------------------------------
// Coalescing (group) window -- the c10d analog of ncclGroupStart/ncclGroupEnd.
// Op-batching only: deferred ops are posted with the same per-tensor tags as
// the non-grouped path (wire-compatible with a non-grouping peer); the single
// win is one stream-sync + one aggregate Work for the whole exchange. IOV
// packing stays governed by COMMUX_COALESCE and is independent of grouping.
// ---------------------------------------------------------------------------

void ProcessGroupUCX::startCoalescing() {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (!group_enabled()) {
    // Grouping disabled: leave coalescing_ false so send()/recv() post
    // immediately and return real per-op Works; endCoalescing() is a no-op.
    coalescing_ = false;
    return;
  }
  coalescing_ = true;
  deferred_.clear();
  pending_ = c10::make_intrusive<UCXWork>(
      worker_, &worker_mu_, c10d::OpType::COALESCED, std::vector<void*>{},
      std::vector<at::Tensor>{});
}

c10::intrusive_ptr<c10d::Work> ProcessGroupUCX::endCoalescing() {
  std::lock_guard<std::mutex> lock(worker_mu_);
  if (!coalescing_) {
    // No open window (grouping disabled): the real per-op Works were already
    // returned by send()/recv(); hand back a completed no-op aggregate.
    return completed_work(c10d::OpType::COALESCED, {}, worker_, &worker_mu_);
  }
  // Close the window and take ownership of its state up front, so that even if
  // a post below throws, the backend is left clean for the next exchange. (The
  // already-posted requests of a failed flush are abandoned -- a post failure
  // is fatal anyway.)
  coalescing_ = false;
  auto deferred = std::move(deferred_);
  deferred_.clear();
  auto work = std::move(pending_);
  pending_.reset();

  // Collect every buffer; one stream-sync covers them all before posting.
  std::vector<at::Tensor> keep;
  for (auto& op : deferred)
    for (auto& t : op.keep) keep.push_back(t);
  sync_batch_streams(keep);

  std::vector<void*> reqs;
  for (auto& op : deferred)
    for (size_t i = 0; i < op.keep.size(); ++i)
      reqs.push_back(
          op.is_send
              ? post_send(op.keep[i], op.peer, op.tag, static_cast<uint32_t>(i),
                          /*collective=*/false)
              : post_recv(op.keep[i], op.peer, op.tag, static_cast<uint32_t>(i),
                          /*collective=*/false));

  static_cast<UCXWork*>(work.get())->adopt(std::move(reqs), std::move(keep));
  return work;
}

}  // namespace commux
