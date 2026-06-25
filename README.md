# commux

[![CI](https://github.com/zoeyzyhu/commux/actions/workflows/ci.yml/badge.svg)](https://github.com/zoeyzyhu/commux/actions/workflows/ci.yml)

A custom **PyTorch `c10d` backend over UCX** that gives real **MPI-style
`(sender, tag)` point-to-point matching** — the thing NCCL cannot do — plus
`allreduce` / `reduce` / `broadcast` / `barrier`, on **CPU host tensors and CUDA
device tensors** (UCX `cuda_ipc` / `cuda_copy` / `gdr_copy`).

Motivation: codes that drive stencil/halo exchange through tagged `c10d`
send/recv (`pg.send(bufs, dst, tag)`) can't use NCCL for it — NCCL ignores tags
and matches only by stream/communicator ordering. commux honors tags via UCX's
`ucp_tag_send_nbx` / `ucp_tag_recv_nbx`.

## Install

commux is **Linux-only** (it builds on UCX, which does not compile on macOS) and
needs **UCX** — but you do **not** install UCX yourself: every wheel **vendors
UCX (+ the gdrcopy userspace lib)**, so a `pip install` is self-contained.

```bash
# Released wheel (self-contained: bundled UCX, no system UCX needed):
pip install commux
pip install "commux @ git+https://github.com/zoeyzyhu/commux"

# From a source checkout -- builds against your active torch, so no isolation:
pip install . --no-build-isolation

# Use a preinstalled / module UCX instead of building one from source:
UCX_ROOT=/path/to/ucx pip install . --no-build-isolation \
  -C cmake.define.COMMUX_UCX_PROVIDER=system

# Force building UCX(+gdrcopy) from source even if a system one exists:
pip install . --no-build-isolation -C cmake.define.COMMUX_UCX_PROVIDER=bundled
```

The installed package under `site-packages/commux/` is fully self-contained:
`_C.so` (the extension), `lib/libcommux.so` plus the vendored `lib/libuc*.so` /
`libgdrapi.so` (relocatable — repointed to `$ORIGIN`), and `include/` with the
commux **and** UCX C++ headers for downstream C++ consumers (see below).

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

Because the wheel ships the C++ **library + headers** (and bundles UCX), a C++
project can just **`pip install commux`** and link the installed package — no
source build, no UCX of its own. Locate it from CMake by probing the Python
package (this is what snapy's `FindCommux.cmake` does):

```cmake
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import commux,os;print(os.path.dirname(commux.__file__))"
  OUTPUT_VARIABLE COMMUX_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_library(COMMUX_LIBRARY commux HINTS ${COMMUX_DIR}/lib)
include_directories(${COMMUX_DIR}/include)  # commux/*.hpp + bundled ucp/*.h
link_directories(${COMMUX_DIR}/lib)         # lets the linker resolve bundled UCX
target_link_libraries(your_lib PUBLIC ${COMMUX_LIBRARY})
```
```cpp
#include <commux/process_group_ucx.hpp>
auto backend = c10::make_intrusive<commux::ProcessGroupUCX>(store, rank, size);
pg->setBackend(c10::DeviceType::CPU,  c10d::ProcessGroup::BackendType::CUSTOM, backend);
pg->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::CUSTOM, backend);
```

Prefer to build from source? commux also exports the CMake target
`commux::commux`, so FetchContent works too:

```cmake
include(FetchContent)
FetchContent_Declare(commux GIT_REPOSITORY https://github.com/zoeyzyhu/commux GIT_TAG main)
FetchContent_MakeAvailable(commux)
target_link_libraries(your_lib PUBLIC commux::commux)
```

## Build the C++ tests

```bash
cmake -S . -B build -DCOMMUX_BUILD_TESTS=ON      # add -DUCX_ROOT=~/ucx-install to use a prebuilt UCX
cmake --build build -j
for r in 0 1; do ./build/test_commux $r 2 127.0.0.1 29581 & done; wait        # CPU
UCXPG_DEVICE=cuda ./build/test_commux 0 2 127.0.0.1 29592 & \
UCXPG_DEVICE=cuda ./build/test_commux 1 2 127.0.0.1 29592 & wait              # CUDA
```

> `test_commux` also covers the coalescing features: it exercises the IOV path
> under `COMMUX_COALESCE=1` and the coalescing window under `COMMUX_GROUP=1`. Set
> `COMMUX_BENCH=1 COMMUX_BENCH_V=<N>` for a `V`-tensor ping-pong latency
> benchmark (compare `COMMUX_COALESCE=0` vs `1`).

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

## Runtime tuning (environment variables)

Read once at process start; opt-in and **off by default**, so default behavior is
unchanged.

| variable | default | meaning |
|----------|---------|---------|
| `COMMUX_COALESCE` | off | Merge a multi-tensor `send()`/`recv()` vector (`tensors.size() > 1`, all on one memory type) into **one** UCX message via the IOV datatype — one rendezvous / tag-match / stream-sync instead of `V`. **Changes the wire format**, so every rank must set it the same way. `1`/`on` to enable. |
| `COMMUX_GROUP` | off | Enable the coalescing window — the c10d `startCoalescing`/`endCoalescing` hooks, i.e. the analog of `ncclGroupStart`/`ncclGroupEnd`. Between the markers, `send`/`recv` defer their posts and are flushed together with a single stream-sync at `endCoalescing`. Wire-compatible with a non-grouping peer. `1`/`on` to enable. |

**Always on (no flag):** wait paths are **event-driven** — a `Work::wait()`
sleeps on the worker's wakeup fd (`UCP_FEATURE_WAKEUP` + `ucp_worker_wait`)
instead of busy-spinning `ucp_worker_progress`, so a rank blocked on a transfer
no longer pins a CPU core. Falls back to a yielding poll if the active transports
expose no wakeup fd.

### When do these help?

- **`COMMUX_COALESCE` (IOV packing)** cuts per-message overhead, so it helps most
  when one `send`/`recv` carries **many** buffers over a **latency-bound /
  high-latency link (InfiniBand)**. Caveat: UCX's IOV path is **not zero-copy for
  CUDA** (it gathers into a staging buffer), so for large vectors on intra-node
  `cuda_ipc` it can be *slower* than separate messages. Measured intra-node:
  faster at `V≈2`, slower at `V≈16` — benchmark before enabling.
- **`COMMUX_GROUP` (op-batching)** collapses an exchange's many per-tensor posts
  into one stream-sync + one aggregate `Work`. Because the posts are already
  issued concurrently, the intra-node win is small; it is most useful on
  multi-node **InfiniBand**, where it reduces handshakes. Exposed through the
  standard c10d coalescing API, so a C++ consumer just calls
  `pg->startCoalescing(dev)` / `pg->endCoalescing(dev)->wait()` around its posts.
  The two flags compose: group the window, and IOV-pack any multi-tensor op in it.

## Design

64-bit `ucp_tag` = `[63:48] senderRank | [47:33] sub-index | [32] collective-bit
| [31:0] userTag`, so receivers match exactly on `(sender, tag)`; `recvAnysource`
wildcards the rank field. Endpoints bootstrap by exchanging worker addresses
through the `c10d::Store`. `send`/`recv` are non-blocking and return a `Work`
whose `wait()` is event-driven — it sleeps on the worker's wakeup fd rather than
busy-spinning `ucp_worker_progress`; collectives run over tagged p2p with
`at::add/minimum/maximum` so the same code reduces CPU and CUDA tensors. CUDA
buffers are stream-synchronized before UCX touches them. Optional
message/operation coalescing is available at runtime — see **Runtime tuning**.
