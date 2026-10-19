#pragma once

namespace mfq {
// File hints follow the operation, not the reader's lifetime. Bulk prefill
// opts in on each read worker; latency-sensitive demand reads use memcpy.
class MappedPrefetchScope {
public:
    explicit MappedPrefetchScope(bool enabled);
    ~MappedPrefetchScope();
    MappedPrefetchScope(const MappedPrefetchScope&)=delete;
    MappedPrefetchScope& operator=(const MappedPrefetchScope&)=delete;
private:
    bool previous_;
};
}
