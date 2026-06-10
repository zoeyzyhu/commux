# commux

A custom **PyTorch `c10d` backend over UCX** that gives real **MPI-style
`(sender, tag)` point-to-point matching** — the thing NCCL cannot do — plus
`allreduce` / `reduce` / `broadcast` / `barrier`, on **CPU host tensors and CUDA
device tensors** (UCX `cuda_ipc` / `cuda_copy` / `gdr_copy`).

Motivation: codes that drive stencil/halo exchange through tagged `c10d`
send/recv (`pg.send(bufs, dst, tag)`) can't use NCCL for it — NCCL ignores tags
and matches only by stream/communicator ordering. commux honors tags via UCX's
`ucp_tag_send_nbx` / `ucp_tag_recv_nbx`.

## Install

commux needs **UCX** (optionally built `--with-gdrcopy` for GPUDirect RDMA). You
do **not** have to install UCX yourself — if no system UCX is found, the build
fetches and builds it (and the gdrcopy userspace lib) automatically.

```bash
# Python package (registers a torch.distributed backend). Build against your
# active torch (no build isolation):
pip install . --no-build-isolation

# Use a preinstalled / module UCX instead of auto-building:
UCX_ROOT=/path/to/ucx pip install . --no-build-isolation

# Force building UCX(+gdrcopy) from source even if one exists:
pip install . --no-build-isolation -C cmake.define.COMMUX_UCX_PROVIDER=bundled
```

> `gdr_copy` only engages at runtime when the `gdrdrv` kernel module is loaded
> (`/dev/gdrdrv`) and an RDMA NIC is present; otherwise UCX uses
> `cuda_copy`/`cuda_ipc`. The kernel module is a host/driver prerequisite that a
> wheel cannot provide.

## Use with torch.distributed

```python
import torch, torch.distributed as dist
import commux; commux.register()          # registers backends "ucx" and "commux"

dist.init_process_group(backend="ucx", init_method="env://")
dist.all_reduce(t)                          # CPU or CUDA
dist.isend(x, dst=1, tag=100); dist.irecv(y, src=0, tag=100)
```

Launch with `torchrun --nproc-per-node=N ...` as usual.

> Out-of-order tag matching must use **non-blocking** `isend`/`irecv` (post all,
> then wait). Blocking out-of-order send/recv deadlocks under the CUDA
> rendezvous protocol, exactly as with real MPI.

## Use from C++ (e.g. snapy)

`commux` exports a CMake target `commux::commux`. Pull it in via FetchContent and
construct the backend directly:

```cmake
include(FetchContent)
FetchContent_Declare(commux GIT_REPOSITORY https://github.com/zoeyzyhu/commux GIT_TAG v0.1.0)
FetchContent_MakeAvailable(commux)
target_link_libraries(your_lib PUBLIC commux::commux)
```
```cpp
#include <commux/process_group_ucx.hpp>
auto backend = c10::make_intrusive<commux::ProcessGroupUCX>(store, rank, size);
pg->setBackend(c10::DeviceType::CPU,  c10d::ProcessGroup::BackendType::CUSTOM, backend);
pg->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::CUSTOM, backend);
```

## Build the C++ tests

```bash
cmake -S . -B build -DCOMMUX_BUILD_TESTS=ON      # add -DUCX_ROOT=~/ucx-install to use a prebuilt UCX
cmake --build build -j
for r in 0 1; do ./build/test_commux $r 2 127.0.0.1 29581 & done; wait        # CPU
UCXPG_DEVICE=cuda ./build/test_commux 0 2 127.0.0.1 29592 & \
UCXPG_DEVICE=cuda ./build/test_commux 1 2 127.0.0.1 29592 & wait              # CUDA
```

## CMake options

| option | default | meaning |
|--------|---------|---------|
| `COMMUX_UCX_PROVIDER` | `auto` | `auto` (system, else build) / `system` / `bundled` |
| `COMMUX_WITH_GDRCOPY` | `auto` | build gdrcopy into bundled UCX (`auto`/`on`/`off`) |
| `COMMUX_UCX_VERSION` | `1.18.0` | UCX release to bundle |
| `COMMUX_CUDA` | `ON` | enable CUDA path if `c10_cuda` is present |
| `COMMUX_BUILD_TESTS` | `OFF` | build `test_commux` |
| `COMMUX_BUILD_PYTHON` | `OFF` | build the `commux._C` extension (set by the wheel) |

`scripts/build_ucx.sh [PREFIX] [UCX_VER] [GDR_VER]` builds UCX+gdrcopy once with
the same flags, for pointing `UCX_ROOT` at.

## Design

64-bit `ucp_tag` = `[63:48] senderRank | [47:33] sub-index | [32] collective-bit
| [31:0] userTag`, so receivers match exactly on `(sender, tag)`; `recvAnysource`
wildcards the rank field. Endpoints bootstrap by exchanging worker addresses
through the `c10d::Store`. `send`/`recv` are non-blocking and return a `Work`
that drives `ucp_worker_progress`; collectives run over tagged p2p with
`at::add/minimum/maximum` so the same code reduces CPU and CUDA tensors. CUDA
buffers are stream-synchronized before UCX touches them.
