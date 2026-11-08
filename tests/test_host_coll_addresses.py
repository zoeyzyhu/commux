"""Host collectives must hand UCX the same few host addresses however often the
caller passes fresh tensors (the xpmem rcache attaches every new remote address
and never evicts, so a fresh address per step leaks one mapping per step).

Run:
    torchrun --standalone --nnodes=1 --nproc-per-node=3 tests/test_host_coll_addresses.py

Builds tests/ucp_send_addr_shim.c against commux's bundled UCX headers, re-runs
itself with it preloaded, then does 100 all_reduces and 20 barriers on fresh CPU
tensors (the previous ones kept alive, so every input has a new address) and
checks the sums and that commux posted host sends from at most 4 addresses.
"""
import ctypes, os, subprocess, sys, tempfile

import commux

here = os.path.dirname(os.path.abspath(__file__))
if "COMMUX_SHIM" not in os.environ:
    shim = os.path.join(tempfile.gettempdir(), "commux_send_addr_shim_%d.so" % os.getpid())
    inc = os.path.join(os.path.dirname(commux.__file__), "include")
    subprocess.check_call(["cc", "-shared", "-fPIC", "-O1", "-I" + inc, "-o", shim,
                           os.path.join(here, "ucp_send_addr_shim.c"), "-ldl"])
    env = dict(os.environ, COMMUX_SHIM=shim, LD_PRELOAD=shim)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import torch
import torch.distributed as dist

commux.register()
dist.init_process_group(backend="ucx", init_method="env://")
rank, world = dist.get_rank(), dist.get_world_size()
lib = ctypes.CDLL(None)
lib.shim_distinct_host_send_addrs.restype = ctypes.c_long
lib.shim_host_sends.restype = ctypes.c_long

keep, bad = [], 0
dist.all_reduce(torch.ones(3, dtype=torch.float64))   # warm-up
lib.shim_reset()
for step in range(100):
    t = torch.full((3,), float(rank + step), dtype=torch.float64)
    keep.append(t)
    dist.all_reduce(t)
    bad += not torch.equal(t, torch.full((3,), float(world * step + world * (world - 1) // 2),
                                         dtype=torch.float64))
for _ in range(20):
    dist.barrier()
addrs, sends = lib.shim_distinct_host_send_addrs(), lib.shim_host_sends()
print(f"rank {rank}: {sends} host sends from {addrs} distinct addresses, {bad} wrong sums", flush=True)
ok = torch.tensor([int(bad == 0 and addrs <= 4)], dtype=torch.int32)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
dist.destroy_process_group()
if not ok.item():
    print("FAIL: host collective sends must reuse a few fixed addresses", flush=True)
    sys.exit(1)
print("host collective send addresses PASSED", flush=True)
