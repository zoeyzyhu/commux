// pybind module commux._C: exposes a factory that builds a commux::ProcessGroupUCX
// so it can be registered with torch.distributed (see commux/__init__.py).
//
// torch calls the registered creator as creator(store, rank, size, timeout) and
// expects a c10d::Backend back; ProcessGroupUCX is one.

#include <chrono>
#include <utility>

#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <pybind11/chrono.h>

#include "commux/process_group_ucx.hpp"

PYBIND11_MODULE(_C, m) {
  m.doc() = "commux: UCX tag-matching c10d backend for torch.distributed";
  m.def(
      "create_backend",
      [](c10::intrusive_ptr<c10d::Store> store, int rank, int size,
         std::chrono::milliseconds /*timeout*/) {
        return c10::static_intrusive_pointer_cast<c10d::Backend>(
            c10::make_intrusive<commux::ProcessGroupUCX>(std::move(store), rank,
                                                         size));
      },
      py::arg("store"), py::arg("rank"), py::arg("size"),
      py::arg("timeout") = std::chrono::milliseconds(0));
}
