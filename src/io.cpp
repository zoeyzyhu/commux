#include "commux/io.hpp"

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <cstring>
#include <stdexcept>

#include "commux/process_group_ucx.hpp"

namespace commux {

namespace {

constexpr int kBroadcastSizeTag = 0x7100;
constexpr int kBroadcastDataTag = 0x7101;
constexpr int kGatherSizeTag = 0x7200;
constexpr int kGatherDataTag = 0x7201;

at::Tensor byte_tensor_from_copy(const void* data, std::size_t nbytes) {
  auto t = torch::empty({static_cast<int64_t>(nbytes)}, torch::kUInt8);
  if (nbytes != 0) std::memcpy(t.data_ptr(), data, nbytes);
  return t;
}

at::Tensor byte_tensor(std::size_t nbytes) {
  return torch::empty({static_cast<int64_t>(nbytes)}, torch::kUInt8);
}

at::Tensor i64_tensor(std::int64_t value) {
  return torch::tensor({value}, torch::TensorOptions().dtype(torch::kInt64));
}

std::int64_t tensor_i64(const at::Tensor& t) {
  return t.item<std::int64_t>();
}

}  // namespace

IOContext::IOContext(c10::intrusive_ptr<c10d::Store> store, int rank, int size)
    : rank_(rank),
      size_(size),
      pg_(c10::make_intrusive<ProcessGroupUCX>(std::move(store), rank, size)) {}

void IOContext::barrier() { pg_->barrier()->wait(); }

void IOContext::send_bytes(const void* data, std::size_t nbytes, int dst,
                           int tag) {
  auto t = byte_tensor_from_copy(data, nbytes);
  std::vector<at::Tensor> tensors{t};
  pg_->send(tensors, dst, tag)->wait();
}

void IOContext::recv_bytes(void* data, std::size_t nbytes, int src, int tag) {
  auto t = byte_tensor(nbytes);
  std::vector<at::Tensor> tensors{t};
  pg_->recv(tensors, src, tag)->wait();
  if (nbytes != 0) std::memcpy(data, t.data_ptr(), nbytes);
}

void IOContext::broadcast_bytes(std::vector<std::uint8_t>& bytes, int root) {
  if (root < 0 || root >= size_)
    throw std::invalid_argument("commux::IOContext broadcast root out of range");

  std::int64_t nbytes = static_cast<std::int64_t>(bytes.size());
  if (rank_ == root) {
    for (int r = 0; r < size_; ++r) {
      if (r == root) continue;
      auto size_t = i64_tensor(nbytes);
      std::vector<at::Tensor> size_vec{size_t};
      pg_->send(size_vec, r, kBroadcastSizeTag)->wait();
      send_bytes(bytes.data(), bytes.size(), r, kBroadcastDataTag);
    }
  } else {
    auto size_t = torch::zeros({1}, torch::kInt64);
    std::vector<at::Tensor> size_vec{size_t};
    pg_->recv(size_vec, root, kBroadcastSizeTag)->wait();
    nbytes = tensor_i64(size_t);
    if (nbytes < 0)
      throw std::runtime_error("commux::IOContext negative broadcast size");
    bytes.resize(static_cast<std::size_t>(nbytes));
    recv_bytes(bytes.data(), bytes.size(), root, kBroadcastDataTag);
  }
  barrier();
}

std::vector<std::uint8_t> IOContext::gather_bytes(
    const std::vector<std::uint8_t>& bytes, int root) {
  if (root < 0 || root >= size_)
    throw std::invalid_argument("commux::IOContext gather root out of range");

  std::vector<std::uint8_t> out;
  if (rank_ == root) {
    std::vector<std::vector<std::uint8_t>> parts(size_);
    parts[root] = bytes;
    for (int r = 0; r < size_; ++r) {
      if (r == root) continue;
      auto size_t = torch::zeros({1}, torch::kInt64);
      std::vector<at::Tensor> size_vec{size_t};
      pg_->recv(size_vec, r, kGatherSizeTag)->wait();
      auto nbytes = tensor_i64(size_t);
      if (nbytes < 0)
        throw std::runtime_error("commux::IOContext negative gather size");
      parts[r].resize(static_cast<std::size_t>(nbytes));
      recv_bytes(parts[r].data(), parts[r].size(), r, kGatherDataTag);
    }
    std::size_t total = 0;
    for (auto const& part : parts) total += part.size();
    out.reserve(total);
    for (auto const& part : parts)
      out.insert(out.end(), part.begin(), part.end());
  } else {
    auto size_t = i64_tensor(static_cast<std::int64_t>(bytes.size()));
    std::vector<at::Tensor> size_vec{size_t};
    pg_->send(size_vec, root, kGatherSizeTag)->wait();
    send_bytes(bytes.data(), bytes.size(), root, kGatherDataTag);
  }
  barrier();
  return out;
}

std::shared_ptr<IOContext> make_tcp_io_context(const std::string& host,
                                               int port, int rank, int size) {
  c10d::TCPStoreOptions opts;
  opts.port = static_cast<std::uint16_t>(port);
  opts.isServer = rank == 0;
  opts.numWorkers = size;
  opts.waitWorkers = true;
  auto store = c10::make_intrusive<c10d::TCPStore>(host, opts);
  return std::make_shared<IOContext>(store, rank, size);
}

}  // namespace commux
