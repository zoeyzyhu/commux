"""End-to-end test of commux through the real torch.distributed API.

Run (2 ranks):
    pip install .                                          # or build_ext --inplace
    torchrun --nnodes=1 --nproc-per-node=2 tests/test_torch_dist.py
    UCXPG_DEVICE=cuda torchrun --nnodes=1 --nproc-per-node=2 tests/test_torch_dist.py

Exercises dist.all_reduce and tagged dist.isend/dist.irecv (the latter delivered
out of order to prove (sender, tag) matching -- the NCCL gap). Out-of-order
matching uses non-blocking isend/irecv (post all, then wait); blocking
send/recv out of order would deadlock under the CUDA rendezvous protocol, as
with real MPI.
"""

import os

import torch
import torch.distributed as dist

import commux

commux.register()

dist.init_process_group(backend="ucx", init_method="env://")
rank = dist.get_rank()
world = dist.get_world_size()
assert dist.get_backend() == "ucx", dist.get_backend()

if os.environ.get("UCXPG_DEVICE") == "cuda" and torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = torch.device("cuda", local_rank % torch.cuda.device_count())
    torch.cuda.set_device(dev)
else:
    dev = torch.device("cpu")
if rank == 0:
    print(f"[rank 0] backend={dist.get_backend()} device={dev}")

# allreduce SUM across ranks (ranks contribute 1..world)
t = torch.full((4,), float(rank + 1), device=dev)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expect = world * (world + 1) / 2.0
assert torch.allclose(t, torch.full((4,), expect, device=dev)), (rank, t, expect)
print(f"[rank {rank}] all_reduce SUM OK -> {t[0].item()}")

# tagged out-of-order point-to-point via non-blocking isend/irecv
if world >= 2:
    if rank == 0:
        reqs = [
            dist.isend(torch.full((3,), 1.0, device=dev), dst=1, tag=100),
            dist.isend(torch.full((3,), 2.0, device=dev), dst=1, tag=200),
        ]
        for r in reqs:
            r.wait()
    elif rank == 1:
        b = torch.zeros(3, device=dev)
        a = torch.zeros(3, device=dev)
        rb = dist.irecv(b, src=0, tag=200)  # request the second-sent msg first
        ra = dist.irecv(a, src=0, tag=100)
        rb.wait()
        ra.wait()
        ok = torch.allclose(b, torch.full((3,), 2.0, device=dev)) and torch.allclose(
            a, torch.full((3,), 1.0, device=dev)
        )
        assert ok, (a, b)
        print("[rank 1] tag-matched isend/irecv via torch.distributed OK")

dist.barrier()
if rank == 0:
    print("ALL torch.distributed commux TESTS PASSED")
dist.destroy_process_group()
