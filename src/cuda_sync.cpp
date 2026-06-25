// CUDA stream-sync shim, isolated into its own shared library
// (libcommux_cuda.so) so that the MAIN libcommux.so carries no link-time
// dependency on libc10_cuda.so. The main library dlopen()s this sibling lazily
// at runtime (see resolve_cuda_sync() in process_group_ucx.cpp): it is present
// in a CUDA torch -- where libc10_cuda.so exists -- and harmlessly absent in a
// CPU-only torch, so one wheel loads on both. This is the only commux object
// that references c10::cuda, hence the only one that needs libc10_cuda.so.
#include <c10/cuda/CUDAStream.h>

// Block until the current torch CUDA stream of `device_index` is idle. UCX
// moves CUDA memory on its own stream, unordered with respect to the torch
// stream that produced/consumes the buffer, so the caller drains that stream
// before handing the buffer to UCX. C ABI on purpose: the main library resolves
// it by name.
extern "C" void commux_cuda_sync_stream(int device_index) {
  c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_index))
      .synchronize();
}
