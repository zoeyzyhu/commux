#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace commux {

class ProcessGroupUCX;

class IOContext {
 public:
  IOContext(c10::intrusive_ptr<c10d::Store> store, int rank, int size);

  int rank() const { return rank_; }
  int size() const { return size_; }

  void barrier();
  void broadcast_bytes(std::vector<std::uint8_t>& bytes, int root);
  std::vector<std::uint8_t> gather_bytes(const std::vector<std::uint8_t>& bytes,
                                         int root);
  void send_bytes(const void* data, std::size_t nbytes, int dst, int tag);
  void recv_bytes(void* data, std::size_t nbytes, int src, int tag);

 private:
  int rank_;
  int size_;
  c10::intrusive_ptr<ProcessGroupUCX> pg_;
};

std::shared_ptr<IOContext> make_tcp_io_context(const std::string& host,
                                               int port, int rank, int size);

}  // namespace commux
