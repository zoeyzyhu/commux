"""An unsupported ReduceOp is rejected on every rank before any message (#13).

Run under a timeout; a hang is a failure:
    timeout 120 torchrun --nnodes=1 --nproc-per-node=4 tests/test_reduce_op.py

reduce (every root) and all_reduce with AVG and BAND must raise on every rank.
If the op were checked only where data is combined, the ranks that just send
would return, and in all_reduce they would then wait for a broadcast that never
comes. A SUM all_reduce after the rejected calls checks that nothing was left
in flight on their tags. Each rank counts its failures and they are agreed with
a final all_reduce. At one rank no message is sent, and the op must still be
rejected.
"""
import sys

import torch
import torch.distributed as dist
import commux

commux.register()
dist.init_process_group(backend="ucx", init_method="env://")
rank = dist.get_rank()
world = dist.get_world_size()
failures = []


def expect_rejected(name, call):
    try:
        call()
    except RuntimeError as e:
        if "unsupported ReduceOp" not in str(e):
            failures.append(f"{name}: rank {rank} raised another error: {e}")
        return
    failures.append(f"{name}: rank {rank} returned without an error")


for op in (dist.ReduceOp.AVG, dist.ReduceOp.BAND):
    for root in range(world):
        t = torch.ones(5, dtype=torch.float64)
        expect_rejected(f"reduce {op} root={root}",
                        lambda: dist.reduce(t, dst=root, op=op))
    t = torch.ones(5, dtype=torch.float64)
    expect_rejected(f"all_reduce {op}", lambda: dist.all_reduce(t, op=op))

t = torch.full((5,), float(rank + 1), dtype=torch.float64)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
want = torch.full((5,), world * (world + 1) / 2.0, dtype=torch.float64)
if not torch.equal(t, want):
    failures.append(f"all_reduce SUM after the rejected calls: rank {rank} "
                    f"holds {t.tolist()}, expected {want.tolist()}")

for msg in failures:
    print(f"FAIL {msg}", flush=True)
count = torch.tensor([float(len(failures))])
dist.all_reduce(count, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"world {world}: {int(count.item())} failure(s)"
          + ("" if count.item() else "; ALL commux reduce-op TESTS PASSED"),
          flush=True)
dist.destroy_process_group()
sys.exit(1 if count.item() else 0)
