// Multi-rank test for commux::ProcessGroupUCX. Bootstraps ranks through a
// c10d::TCPStore (rank 0 is the store server), then exercises the two
// capabilities that motivate this backend:
//
//   1. Tag-matched point-to-point: rank 1 receives tag B *before* tag A while
//      rank 0 sends A then B. Correct delivery proves real (sender,tag) matching
//      -- a case NCCL gets wrong because it ignores tags and matches by order.
//      Posted non-blocking (post all, then wait) so it is deadlock-free under
//      the rendezvous protocol CUDA uses.
//   2. allreduce SUM/MIN/MAX, reduce-to-root, barrier.
//
// Usage:
//   test_commux <rank> <world_size> [master_host=127.0.0.1] [master_port=29555]
//   UCXPG_DEVICE=cuda  runs the same checks on GPU tensors.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <c10/cuda/CUDAFunctions.h>

#include "commux/process_group_ucx.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <rank> <world_size> [host=127.0.0.1] [port=29555]\n",
                 argv[0]);
    return 2;
  }
  int rank = std::atoi(argv[1]);
  int size = std::atoi(argv[2]);
  std::string host = argc > 3 ? argv[3] : "127.0.0.1";
  int port = argc > 4 ? std::atoi(argv[4]) : 29555;

  c10d::TCPStoreOptions sopts;
  sopts.port = static_cast<std::uint16_t>(port);
  sopts.isServer = (rank == 0);
  sopts.numWorkers = size;
  sopts.waitWorkers = true;
  auto store = c10::make_intrusive<c10d::TCPStore>(host, sopts);

  auto pg = c10::make_intrusive<commux::ProcessGroupUCX>(store, rank, size);

  const char* dev_env = std::getenv("UCXPG_DEVICE");
  bool want_cuda = dev_env != nullptr && std::string(dev_env) == "cuda";
  torch::Device device(torch::kCPU);
  if (want_cuda) {
    if (!torch::cuda::is_available()) {
      std::fprintf(stderr, "[rank %d] CUDA requested but not available\n", rank);
      return 2;
    }
    int dev_index = rank % static_cast<int>(torch::cuda::device_count());
    device = torch::Device(torch::kCUDA, dev_index);
    c10::cuda::set_device(dev_index);
  }
  std::printf("[rank %d/%d] commux process group up (backend=%s, device=%s)\n",
              rank, size, pg->getBackendName().c_str(),
              want_cuda ? device.str().c_str() : "cpu");

  auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  // --- Test 1: tag-matched p2p between ranks 0 and 1 (non-blocking) --------
  const int TAG_A = 100;
  const int TAG_B = 200;
  if (size >= 2 && rank == 0) {
    auto a = torch::full({4}, 1.0, f64);
    auto b = torch::full({4}, 2.0, f64);
    std::vector<at::Tensor> va{a}, vb{b};
    auto wa = pg->send(va, /*dst=*/1, TAG_A);  // post A ...
    auto wb = pg->send(vb, /*dst=*/1, TAG_B);  // ... then B, both in flight
    wa->wait();
    wb->wait();
    std::printf("[rank 0] sent tag %d (=1.0) then tag %d (=2.0)\n", TAG_A,
                TAG_B);
  } else if (size >= 2 && rank == 1) {
    auto bufB = torch::zeros({4}, f64);
    auto bufA = torch::zeros({4}, f64);
    std::vector<at::Tensor> vB{bufB}, vA{bufA};
    auto wB = pg->recv(vB, /*src=*/0, TAG_B);  // request B first ...
    auto wA = pg->recv(vA, /*src=*/0, TAG_A);  // ... then A
    wB->wait();
    wA->wait();
    bool ok = bufB.eq(2.0).all().item<bool>() && bufA.eq(1.0).all().item<bool>();
    check(ok, "tag-matched p2p: out-of-order recv delivered by tag, not order");
  }

  pg->barrier()->wait();

  // --- Test 2: allreduce SUM / MIN / MAX ----------------------------------
  {
    double mine = static_cast<double>(rank + 1);
    double expect_sum = static_cast<double>(size) * (size + 1) / 2.0;
    auto run = [&](c10d::ReduceOp::RedOpType op) {
      auto t = torch::full({3}, mine, f64);
      std::vector<at::Tensor> v{t};
      c10d::AllreduceOptions o;
      o.reduceOp = op;
      pg->allreduce(v, o)->wait();
      return t[0].item<double>();
    };
    check(run(c10d::ReduceOp::SUM) == expect_sum, "allreduce SUM");
    check(run(c10d::ReduceOp::MIN) == 1.0, "allreduce MIN");
    check(run(c10d::ReduceOp::MAX) == static_cast<double>(size), "allreduce MAX");
  }

  // --- Test 3: reduce SUM to root 0 ---------------------------------------
  {
    auto t = torch::full({2}, static_cast<double>(rank + 1), f64);
    std::vector<at::Tensor> v{t};
    c10d::ReduceOptions o;
    o.reduceOp = c10d::ReduceOp::SUM;
    o.rootRank = 0;
    pg->reduce(v, o)->wait();
    if (rank == 0) {
      double expect = static_cast<double>(size) * (size + 1) / 2.0;
      check(t[0].item<double>() == expect, "reduce SUM lands on root");
    }
  }

  pg->barrier()->wait();
  if (rank == 0) {
    std::printf(g_failures == 0 ? "\nALL TESTS PASSED\n"
                                : "\n%d TEST(S) FAILED\n",
                g_failures);
  }
  return g_failures == 0 ? 0 : 1;
}
