#pragma once

// commux: a custom PyTorch c10d::Backend implemented over UCX (Unified
// Communication X).
//
// Why this exists: codes that drive halo/stencil exchange through tagged
// point-to-point send/recv (c10d send/recv carry an MPI-style `tag`) cannot use
// NCCL for it -- NCCL ignores tags and matches only by stream/communicator
// ordering. commux uses UCX's tag-matching API (ucp_tag_send_nbx /
// ucp_tag_recv_nbx) to give real MPI-style (sender, tag) matching, plus
// allreduce/reduce/broadcast/barrier built on top of tagged p2p. It works for
// both CPU host tensors and CUDA device tensors (UCX cuda_copy / cuda_ipc /
// gdr_copy).

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace commux {

// 64-bit ucp_tag_t layout, chosen so a receiver matches on (senderRank, tag)
// exactly -- the MPI-style behavior NCCL cannot provide:
//
//   bits [63:48] (16)  sender rank
//   bits [47:33] (15)  sub-index (tensor position within a std::vector arg)
//   bit  [32]          collective namespace (keeps collective tags disjoint
//                      from halo p2p tags)
//   bits [31:0]  (32)  user tag
//
ucp_tag_t make_ucp_tag(int sender_rank, uint32_t user_tag, uint32_t sub_index,
                       bool collective);

// Tag mask that wildcards the sender-rank field, used by recvAnysource.
constexpr ucp_tag_t kTagMaskAnySource = 0x0000FFFFFFFFFFFFull;

class ProcessGroupUCX : public c10d::Backend {
 public:
  ProcessGroupUCX(c10::intrusive_ptr<c10d::Store> store, int rank, int size);
  ~ProcessGroupUCX() override;

  const std::string getBackendName() const override { return "ucx"; }

  // --- point-to-point (tagged) -------------------------------------------
  c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor>& tensors,
                                      int dstRank, int tag) override;
  c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor>& tensors,
                                      int srcRank, int tag) override;
  c10::intrusive_ptr<c10d::Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                               int tag) override;

  // --- collectives (built on tagged p2p) ---------------------------------
  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;
  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

 private:
  void init_ucx();
  void bootstrap_endpoints();

  // Post a non-blocking tagged send/recv. Returns a ucp request pointer, or
  // nullptr if the operation completed inline. Throws on immediate error.
  // Caller must hold worker_mu_.
  void* post_send(const at::Tensor& t, int dst, uint32_t user_tag,
                  uint32_t sub_index, bool collective);
  void* post_recv(at::Tensor& t, int src, uint32_t user_tag,
                  uint32_t sub_index, bool collective);

  // Blocking variants used by the collective algorithms. Caller holds the lock.
  void coll_send(const at::Tensor& t, int dst, uint32_t user_tag);
  void coll_recv(at::Tensor& t, int src, uint32_t user_tag);
  void wait_request(void* req);  // drive progress until one request completes

  // Collective primitives that assume worker_mu_ is held.
  void reduce_locked(std::vector<at::Tensor>& tensors, c10d::ReduceOp op,
                     int root);
  void broadcast_locked(std::vector<at::Tensor>& tensors, int root);

  c10::intrusive_ptr<c10d::Store> store_;
  int rank_;
  int size_;

  ucp_context_h context_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  std::vector<ucp_ep_h> eps_;  // one endpoint per peer rank (eps_[rank_] unused)

  // Serializes all access to the single ucp worker (post + progress + wait).
  std::mutex worker_mu_;
};

}  // namespace commux
