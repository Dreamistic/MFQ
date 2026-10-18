#include "moe_host_expert_cache.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {
struct Ledger {
    std::atomic<std::size_t> managed{0},managed_peak{0},transient{0},transient_peak{0};
};
void charge(std::atomic<std::size_t>& used,std::atomic<std::size_t>& peak,std::size_t bytes) {
    const auto value=used.fetch_add(bytes)+bytes;
    auto old=peak.load();
    while (old<value && !peak.compare_exchange_weak(old,value)) {}
}
}

struct MoeHostExpertReservation {
    std::shared_ptr<Ledger> ledger;
    std::size_t bytes;
    bool managed;
    MoeHostExpertReservation(std::shared_ptr<Ledger> value,std::size_t size,bool retained)
        :ledger(std::move(value)),bytes(size),managed(retained) {
        if (managed) charge(ledger->managed,ledger->managed_peak,bytes);
        else charge(ledger->transient,ledger->transient_peak,bytes);
    }
    ~MoeHostExpertReservation() {
        (managed ? ledger->managed : ledger->transient).fetch_sub(bytes);
    }
};

struct MoeHostExpertCache::Impl {
    struct Entry { Lease value; std::list<mfq::MoeCacheKey>::iterator position; bool reused=false; };
    struct Flight {
        bool ready=false;
        bool retain=true;
        Lease value;
        std::exception_ptr error;
        std::condition_variable condition;
    };
    const std::size_t budget;
    std::shared_ptr<Ledger> ledger=std::make_shared<Ledger>();
    mutable std::mutex mutex;
    std::list<mfq::MoeCacheKey> lru;
    std::list<mfq::MoeCacheKey> reused;
    std::unordered_map<mfq::MoeCacheKey,Entry,mfq::MoeCacheKeyHash> entries;
    std::unordered_map<mfq::MoeCacheKey,std::shared_ptr<Flight>,mfq::MoeCacheKeyHash> flights;
    // Bypassed payloads still have an active consumer. Reuse that immutable
    // lease even when the RAM budget cannot retain it in the LRU.
    std::unordered_map<mfq::MoeCacheKey,std::weak_ptr<const MoeHostExpert>,mfq::MoeCacheKeyHash> transient;
    std::unordered_set<mfq::MoeCacheKey,mfq::MoeCacheKeyHash> exchanging;
    std::condition_variable exchange_complete;
    std::size_t exchange_placeholders=0;
    MoeHostExpertCacheStats counters;
    bool preserve=false,reuse_learning=true;

    explicit Impl(std::size_t bytes):budget(bytes) {}
    void wait_exchange(std::unique_lock<std::mutex>& lock,mfq::MoeCacheKey key) {
        if(exchanging.contains(key))exchange_complete.wait(lock,[&]{return !exchanging.contains(key);});
    }
    void remove(decltype(entries)::iterator found) {
        counters.resident_bytes-=found->second.value->bytes;
        if(found->first.source<0)counters.optional_bytes-=found->second.value->bytes;
        if(found->second.reused)counters.reuse_protected_bytes-=found->second.value->bytes;
        (found->second.reused?reused:lru).erase(found->second.position);
        entries.erase(found);
    }
    void touch(mfq::MoeCacheKey key,Entry& entry) {
        if(!reuse_learning)return;
        if(!counters.reuse_priority || key.source<0) {
            lru.splice(lru.begin(),lru,entry.position);return;
        }
        reused.splice(reused.begin(),entry.reused?reused:lru,entry.position);
        if(!entry.reused) {
            entry.reused=true;counters.reuse_protected_bytes+=entry.value->bytes;++counters.reuse_promotions;
        }
        const auto limit=budget-budget/4;
        while(counters.reuse_protected_bytes>limit && !reused.empty()) {
            auto candidate=reused.end();
            do {--candidate;}while(candidate!=reused.begin() && exchanging.contains(*candidate));
            if(exchanging.contains(*candidate))break;
            auto& old=entries.at(*candidate);
            counters.reuse_protected_bytes-=old.value->bytes;old.reused=false;
            lru.splice(lru.begin(),reused,old.position);
        }
    }
    bool make_room(std::size_t bytes) {
        if (!bytes || bytes>budget) return false;
        for(auto* queue:{&lru,&reused}) {
          auto candidate=queue->end();
          while (ledger->managed.load()>budget-bytes && candidate!=queue->begin()) {
            --candidate;
            if(exchanging.contains(*candidate))continue;
            auto found=entries.find(*candidate);
            // Negative source IDs are reusable prefill backup images. Their
            // GPU copy still exists outside the loan; discard them before a
            // request needs more RAM, even in complete-residency mode.
            if (preserve && candidate->source>=0) continue;
            // Externally held payloads stay charged even after removal. Keep
            // their lookup entry rather than evicting useful pinned data.
            if (found->second.value.use_count()!=1) continue;
            candidate=queue->erase(candidate);
            counters.resident_bytes-=found->second.value->bytes;
            if(found->first.source<0)counters.optional_bytes-=found->second.value->bytes;
            if(found->second.reused)counters.reuse_protected_bytes-=found->second.value->bytes;
            entries.erase(found);
            ++counters.evictions;
          }
          if(ledger->managed.load()<=budget-bytes)return true;
        }
        return false;
    }
    void publish(mfq::MoeCacheKey key,Lease value) {
        lru.push_front(key);
        try {
            if (!entries.emplace(key,Entry{value,lru.begin()}).second)
                throw std::logic_error("duplicate host expert publication");
        }
        catch (...) { lru.pop_front(); throw; }
        counters.resident_bytes+=value->bytes;
        if(key.source<0)counters.optional_bytes+=value->bytes;
    }
};

MoeHostExpertCache::MoeHostExpertCache(std::size_t bytes):impl_(std::make_unique<Impl>(bytes)) {}
MoeHostExpertCache::~MoeHostExpertCache()=default;
void MoeHostExpertCache::enable_reuse_priority() {
    std::lock_guard lock(impl_->mutex);impl_->counters.reuse_priority=true;
}
void MoeHostExpertCache::set_reuse_learning(bool enabled) {
    std::lock_guard lock(impl_->mutex);impl_->reuse_learning=enabled;
}

MoeHostExpertCache::Lease MoeHostExpertCache::find(mfq::MoeCacheKey key,bool touch) {
    std::unique_lock lock(impl_->mutex);impl_->wait_exchange(lock,key);
    const auto found=impl_->entries.find(key);
    if(found==impl_->entries.end())return {};
    if(touch)impl_->touch(key,found->second);
    ++impl_->counters.hits;
    return found->second.value;
}

MoeHostExpertCache::Lease MoeHostExpertCache::try_find(mfq::MoeCacheKey key,bool touch) {
    std::unique_lock lock(impl_->mutex,std::try_to_lock);
    if(!lock.owns_lock() || impl_->exchanging.contains(key))return {};
    const auto found=impl_->entries.find(key);
    if(found==impl_->entries.end())return {};
    if(touch)impl_->touch(key,found->second);
    ++impl_->counters.hits;
    return found->second.value;
}

void MoeHostExpertCache::preserve_resident_experts() {
    std::lock_guard lock(impl_->mutex);
    impl_->preserve=true;
}

MoeHostExpertCache::Lease MoeHostExpertCache::acquire(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load,bool retain) {
    return this->load(key,bytes,load,retain,false);
}

MoeHostExpertCache::Lease MoeHostExpertCache::acquire_scan(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load) {
    return this->load(key,bytes,load,true,true);
}

MoeHostExpertCache::Lease MoeHostExpertCache::load(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load,bool retain,bool scan) {
    if (!load || !bytes) throw std::invalid_argument("host expert needs a loader and exact byte size");
    auto& state=*impl_;
    std::shared_ptr<Impl::Flight> flight;
    std::shared_ptr<MoeHostExpertReservation> reservation;
    bool managed=false;
    {
        std::unique_lock lock(state.mutex);
        state.wait_exchange(lock,key);
        if (retain) {
            const auto hit=state.entries.find(key);
            if (hit!=state.entries.end()) {
                if(!scan)state.touch(key,hit->second);
                ++state.counters.hits;
                return hit->second.value;
            }
            if(const auto active=state.transient.find(key);active!=state.transient.end()) {
                if(auto value=active->second.lock()) {++state.counters.coalesced;return value;}
                state.transient.erase(active);
            }
            const auto pending=state.flights.find(key);
            if (pending!=state.flights.end()) {
                flight=pending->second;
                ++state.counters.coalesced;
                flight->condition.wait(lock,[&] { return flight->ready; });
                if (flight->error) std::rethrow_exception(flight->error);
                return flight->value;
            }
        }
        if(scan && (bytes>state.budget || state.ledger->managed.load()>state.budget-bytes))return {};
        ++state.counters.misses;
        managed=retain && (scan || state.make_room(bytes));
        if (!managed) ++state.counters.bypasses;
        reservation=std::make_shared<MoeHostExpertReservation>(state.ledger,bytes,managed);
        if (retain) {
            flight=std::make_shared<Impl::Flight>();
            state.flights.emplace(key,flight);
        }
    }
    try {
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=std::move(reservation);
        value->bytes=bytes;
        value->weights=load();
        std::lock_guard lock(state.mutex);
        if (managed && (!flight || flight->retain)) {
            state.publish(key,value);
            if(scan){++state.counters.scan_admissions;state.counters.scan_admitted_bytes+=bytes;}
        }
        else if(retain) state.transient.insert_or_assign(key,value);
        if (flight) {
            flight->value=value;
            flight->ready=true;
            state.flights.erase(key);
            flight->condition.notify_all();
        }
        return value;
    } catch (...) {
        if (flight) {
            std::lock_guard lock(state.mutex);
            flight->error=std::current_exception();
            flight->ready=true;
            state.flights.erase(key);
            flight->condition.notify_all();
        }
        throw;
    }
}

bool MoeHostExpertCache::demote(mfq::MoeCacheKey key,std::size_t bytes,const Load& copy) {
    if (!copy || !bytes) throw std::invalid_argument("GPU expert demotion needs exact fields");
    auto& state=*impl_;
    std::shared_ptr<MoeHostExpertReservation> reservation;
    std::shared_ptr<Impl::Flight> flight;
    {
        std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
        if (state.entries.contains(key)) return true;
        if (state.flights.contains(key) || !state.make_room(bytes)) return false;
        reservation=std::make_shared<MoeHostExpertReservation>(state.ledger,bytes,true);
        flight=std::make_shared<Impl::Flight>();
        state.flights.emplace(key,flight);
    }
    try {
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=std::move(reservation);
        value->bytes=bytes;
        value->weights=copy();
        std::lock_guard lock(state.mutex);
        if (flight->retain) {
            state.publish(key,value);
            ++state.counters.gpu_demotions;
        }
        flight->value=value;
        flight->ready=true;
        state.flights.erase(key);
        flight->condition.notify_all();
        return flight->retain;
    } catch (...) {
        std::lock_guard lock(state.mutex);
        flight->error=std::current_exception();
        flight->ready=true;
        state.flights.erase(key);
        flight->condition.notify_all();
        throw;
    }
}

void MoeHostExpertCache::promote(mfq::MoeCacheKey key) {
    auto& state=*impl_;
    std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) { state.remove(found); ++state.counters.promotions; }
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

void MoeHostExpertCache::erase(mfq::MoeCacheKey key) {
    auto& state=*impl_;
    std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) state.remove(found);
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

bool MoeHostExpertCache::contains(mfq::MoeCacheKey key) const {
    std::unique_lock lock(impl_->mutex);impl_->wait_exchange(lock,key);
    return impl_->entries.contains(key);
}

std::size_t MoeHostExpertCache::reclaim_optional(std::size_t bytes) {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    std::size_t released=0;
    auto candidate=state.lru.end();
    while(candidate!=state.lru.begin() && released<bytes && state.counters.optional_bytes) {
        --candidate;
        if(candidate->source>=0)continue;
        if(state.exchanging.contains(*candidate))continue;
        const auto found=state.entries.find(*candidate);
        if(found->second.value.use_count()!=1)continue;
        released+=found->second.value->bytes;
        candidate=state.lru.erase(candidate);
        state.counters.resident_bytes-=found->second.value->bytes;
        state.counters.optional_bytes-=found->second.value->bytes;
        state.entries.erase(found);
        ++state.counters.evictions;
    }
    state.counters.optional_reclaimed_bytes+=released;
    return released;
}
bool MoeHostExpertCache::exchange(mfq::MoeCacheKey incoming,mfq::MoeCacheKey outgoing,
        const Lease& expected,const std::function<MixedMoePool(const MixedMoePool&)>& copy) {
    if(!expected || !copy || incoming==outgoing)return false;
    auto& state=*impl_;std::unique_lock lock(state.mutex);
    state.wait_exchange(lock,incoming);state.wait_exchange(lock,outgoing);
    auto found=state.entries.find(incoming);
    if(found==state.entries.end() || found->second.value!=expected || expected.use_count()!=2 ||
        state.entries.contains(outgoing) || state.flights.contains(incoming) || state.flights.contains(outgoing))return false;
    auto value=std::make_shared<MoeHostExpert>();
    value->reservation=expected->reservation;value->bytes=expected->bytes;
    const auto position=found->second.position;
    const bool was_reused=found->second.reused;
    // Allocate the new lookup entry before copying into the reused fields.
    state.entries.emplace(outgoing,Impl::Entry{value,position});
    try {value->weights=copy(expected->weights);}
    catch(...) {state.entries.erase(outgoing);throw;}
    state.entries.erase(incoming);*position=outgoing;
    if(incoming.source<0)state.counters.optional_bytes-=value->bytes;
    if(outgoing.source<0)state.counters.optional_bytes+=value->bytes;
    if(was_reused)state.counters.reuse_protected_bytes-=value->bytes;
    state.lru.splice(state.lru.begin(),was_reused?state.reused:state.lru,position);
    ++state.counters.promotions;++state.counters.gpu_demotions;
    return true;
}

bool MoeHostExpertCache::exchange_batch(const std::vector<Exchange>& changes,
        const std::function<void()>& publish,const std::function<void()>& prepare,bool independent_readers) {
    if(changes.empty())return false;
    auto& state=*impl_;std::unique_lock lock(state.mutex);
    const auto* option=std::getenv("MFQ_MOE_KEYED_EXCHANGE");
    const bool keyed=option?option[0]=='1':independent_readers;
    std::unordered_set<mfq::MoeCacheKey,mfq::MoeCacheKeyHash> keys;
    struct Prepared {std::shared_ptr<MoeHostExpert> value;std::list<mfq::MoeCacheKey>::iterator position;bool reused;};
    std::vector<Prepared> prepared;prepared.reserve(changes.size());
    for(const auto& change:changes) {
        if(!change.expected || !*change.expected || !change.copy || !change.restore ||
            !keys.insert(change.incoming).second || !keys.insert(change.outgoing).second)return false;
        const auto found=state.entries.find(change.incoming);
        if(found==state.entries.end() || found->second.value!=*change.expected ||
            change.expected->use_count()!=2 || state.entries.contains(change.outgoing) ||
            state.exchanging.contains(change.incoming) || state.exchanging.contains(change.outgoing) ||
            state.flights.contains(change.incoming) || state.flights.contains(change.outgoing))return false;
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=(*change.expected)->reservation;value->bytes=(*change.expected)->bytes;
        prepared.push_back({std::move(value),found->second.position,found->second.reused});
    }
    // The optional keyed path excludes readers of both names while DMA and
    // publication run, allowing unrelated immutable experts to be acquired.
    std::size_t allocated=0,touched=0;
    bool reserved=false;
    const auto unreserve=[&] {
        if(keyed)for(const auto& key:keys)state.exchanging.erase(key);
        if(reserved)state.exchange_placeholders-=changes.size();
    };
    try {
        for(;allocated<changes.size();++allocated)
            state.entries.emplace(changes[allocated].outgoing,
                Impl::Entry{prepared[allocated].value,prepared[allocated].position});
        if(keyed) {
            state.exchanging.reserve(state.exchanging.size()+keys.size());
            for(const auto& key:keys)state.exchanging.insert(key);
            reserved=true;state.exchange_placeholders+=changes.size();
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "moe_host_keyed_exchange scope=first_transaction enabled=1 changes=%zu\n",changes.size());
            lock.unlock();
        }
        if(prepare) {
            touched=changes.size();
            prepare();
        }
        for(std::size_t i=0;i<changes.size();++i) {
            if(!prepare)touched=i+1; // A callback can fail after changing part of a field.
            prepared[i].value->weights=changes[i].copy((*changes[i].expected)->weights);
        }
        if(publish)publish();
    } catch(...) {
        if(!lock.owns_lock())lock.lock();
        const auto failure=std::current_exception();std::exception_ptr recovery;
        while(touched) {
            --touched;
            try {changes[touched].restore((*changes[touched].expected)->weights);}
            catch(...) {if(!recovery)recovery=std::current_exception();}
        }
        for(std::size_t i=0;i<allocated;++i)state.entries.erase(changes[i].outgoing);
        unreserve();lock.unlock();state.exchange_complete.notify_all();
        if(recovery)std::rethrow_exception(recovery);
        std::rethrow_exception(failure);
    }
    if(!lock.owns_lock())lock.lock();
    for(std::size_t i=0;i<changes.size();++i) {
        state.entries.erase(changes[i].incoming);*prepared[i].position=changes[i].outgoing;
        if(changes[i].incoming.source<0)state.counters.optional_bytes-=prepared[i].value->bytes;
        if(changes[i].outgoing.source<0)state.counters.optional_bytes+=prepared[i].value->bytes;
        if(prepared[i].reused)state.counters.reuse_protected_bytes-=prepared[i].value->bytes;
        state.lru.splice(state.lru.begin(),prepared[i].reused?state.reused:state.lru,prepared[i].position);
    }
    state.counters.promotions+=changes.size();state.counters.gpu_demotions+=changes.size();
    unreserve();lock.unlock();state.exchange_complete.notify_all();
    return true;
}

MoeHostExpertCacheStats MoeHostExpertCache::stats() const {
    const auto& state=*impl_;
    std::lock_guard lock(state.mutex);
    auto result=state.counters;
    result.budget_bytes=state.budget;
    result.resident_entries=state.entries.size()-state.exchange_placeholders;
    result.managed_bytes=state.ledger->managed.load();
    result.managed_peak_bytes=state.ledger->managed_peak.load();
    result.transient_bytes=state.ledger->transient.load();
    result.transient_peak_bytes=state.ledger->transient_peak.load();
    return result;
}
