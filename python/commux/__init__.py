"""commux: a UCX tag-matching backend for torch.distributed.

Gives real MPI-style ``(sender, tag)`` point-to-point matching (which NCCL
lacks) plus allreduce/reduce/broadcast/barrier, on CPU and CUDA.

Usage::

    import commux
    commux.register()                       # registers backends "ucx" and "commux"
    torch.distributed.init_process_group(backend="ucx", init_method="env://")

Or hand the resulting process group to a C++ consumer (e.g. snapy) via its
``set_process_group(...)`` hook.
"""

import os

# Make the bundled libcommux.so / libuc*.so next to this package resolvable
# before importing the extension.
_libdir = os.path.join(os.path.dirname(__file__), "lib")
if os.path.isdir(_libdir):
    os.environ["LD_LIBRARY_PATH"] = (
        _libdir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )

import torch.distributed as dist  # noqa: E402

from . import _C  # noqa: E402  (compiled extension)

__all__ = ["register", "create_backend", "preferred_backend"]

_REGISTERED = False


def create_backend(store, rank, size, timeout):
    """torch.distributed creator: returns a c10d::Backend (ProcessGroupUCX)."""
    return _C.create_backend(store, rank, size, timeout)


def register(names=("ucx", "commux")):
    """Register commux with torch.distributed under the given backend name(s).

    Idempotent. After this, ``backend="ucx"`` (or ``"commux"``) works anywhere
    torch.distributed expects a backend.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    for name in names:
        dist.Backend.register_backend(
            name, create_backend, devices=["cpu", "cuda"]
        )
    _REGISTERED = True


def preferred_backend(name="ucx"):
    """Return the c10d backend name to use on this platform, registering commux.

    commux installs only where UCX is supported (Linux), so if you can import it
    and call this, UCX is available -- on CPU **and** CUDA. It registers commux
    and returns ``"ucx"``. The portable pattern is to guard the import and fall
    back to ``"gloo"`` where commux is absent (e.g. macOS)::

        try:
            from commux import preferred_backend
            backend = preferred_backend()   # registers commux -> "ucx"
        except ImportError:
            backend = "gloo"                # macOS / no UCX
        dist.init_process_group(backend=backend, init_method="env://")

    This is how UCX replaces Gloo on CPU where UCX exists, while keeping Gloo as
    the fallback elsewhere.
    """
    register()
    return name
