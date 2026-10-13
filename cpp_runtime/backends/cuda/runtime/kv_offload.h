#pragma once

#include "mfq_tensor_backend.h"
#include "mfq/qsa_kv_store.h"
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <functional>

namespace mfq::cuda {
struct KvOffloadConfig {
    std::size_t gpu_budget_bytes = 0;
    std::size_t ram_budget_bytes = 0;
    std::size_t buffer_bytes = 64ULL << 20;
    std::string directory;
    bool preallocate = false;
    bool ssd_enabled = true;
};

// Immutable microblocks have one native GPU hot copy. Eviction transfers to
// the shared RAM LRU; only RAM eviction writes the session-owned SSD file.
class KvOffloadStore {
public:
    struct Block;
    using BlockPtr = std::shared_ptr<Block>;
    explicit KvOffloadStore(KvOffloadConfig config);
    ~KvOffloadStore();
    BlockPtr write(const mfq_tensor_backend::Tensor& rows);
    mfq_tensor_backend::Tensor read(const BlockPtr& block);
    const KvOffloadConfig& config() const;
    std::vector<std::pair<std::string, double>> metrics() const;
    void flush();
    void set_ram_reservation(std::function<void(std::size_t)> reserve);
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class KvOffloadSequence {
public:
    KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows,
        mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16);
    int64_t position() const { return position_; }
    void reset();
    void truncate(int64_t position);
    void append(const mfq_tensor_backend::Tensor& rows);
    mfq_tensor_backend::Tensor gather(const std::vector<int64_t>& rows) const;
    mfq_tensor_backend::Tensor range(int64_t start, int64_t count) const;
private:
    std::shared_ptr<KvOffloadStore> store_;
    int64_t width_, block_rows_, position_ = 0;
    mfq_tensor_backend::ScalarType dtype_;
    std::vector<KvOffloadStore::BlockPtr> blocks_;
};
}
