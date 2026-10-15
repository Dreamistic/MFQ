#pragma once
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include "dma_copy_batch.h"
#include <array>
#include <list>
#include <map>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <memory>
#include <limits>

namespace mfq::cuda {
// Cache immutable expert fields in pages of KV storage that have never held
// live KV. Ordinary consumers copy into fixed decode staging addresses. The
// phased decode window may read a pinned field directly until the following
// route publication; next_packet() is called only after that reader boundary.
// KV growth is prepared before a window or at its ordered KV service boundary.
class KvExpertLoan {
    struct ProjectionLayout {
        std::array<std::size_t,8> bytes{},offsets{};
        int fields=0;std::size_t logical_bytes=0;
    };
public:
    using Key=std::array<int64_t,3>; // source, expert, field
    static constexpr std::size_t page=64<<10;
    std::size_t add(mfq_tensor_backend::Tensor owner,int64_t capacity,std::size_t row_bytes,bool prefix,int64_t ratio=1,bool manual=false) {
        std::lock_guard<std::mutex> lock(mutex_);
        Region region;region.owner=std::move(owner);region.capacity=capacity;region.row_bytes=row_bytes;region.prefix=prefix;region.ratio=ratio;
        region.manual_live=manual?0:-1;
        region.end=region.owner.nbytes()/page;region.largest_free=region.end;if(region.end)region.free[0]=region.end;
        regions_.push_back(std::move(region));update_bounds();
        initialized_.record(current_stream().stream());
        initialization_waited_=false;
        return regions_.size()-1;
    }
    void ensure(int64_t tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(tokens<=high_water_)return;high_water_=tokens;
        // Most token appends stay inside the same 64 KiB protection pages.
        // Only changed page boundaries can invalidate a cached expert.
        if(update_bounds())revoke();
    }
    void protect(std::size_t region,int64_t rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& r=regions_.at(region);if(rows<=r.manual_live)return;
        r.manual_live=rows;if(update_bounds())revoke();
    }
    void next_packet() {
        std::lock_guard<std::mutex> lock(mutex_);
        if(++packet_==0){for(auto& e:lru_)e.packet=0;packet_=1;}
        // The next route publication fences readers of the previous packet,
        // exactly as for ordinary LRU eviction. Retirement itself never frees
        // an address that an already submitted graph can still be reading.
        while(!retired_.empty()){retired_bytes_+=retired_.front().bytes;erase(retired_.begin());}
        // The previous route's consumers have finished before this boundary.
        // Publish only now: an allocated projection may still have DMA pending
        // during its first use and must not enter the next prepare's hot set.
        for(const auto& key:pending_publications_)if(auto found=entries_.find(key);found!=entries_.end()) {
            auto& entry=*found->second;
            if(!entry.retired && !entry.published){entry.published=true;notify_projection(entry,true);}
        }
        pending_publications_.clear();
    }
    // Pin every known member of this packet before an allocation can evict a
    // later routed projection. Actual pin() calls still update LRU recency.
    void protect_packet_fields(const std::vector<Key>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& key:keys)if(auto found=entries_.find(key);found!=entries_.end())
            found->second->packet=packet_;
    }
    using ProjectionFill=std::function<void(void*,const std::array<std::size_t,8>&)>;
    using ProjectionObserver=std::function<void(int64_t,const void*,const std::array<std::size_t,8>&)>;
    void observe_projections(int64_t source,ProjectionObserver observer) {
        if(!observer)throw std::invalid_argument("empty KV projection observer");
        std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& entry:lru_)if(entry.key[0]==source && entry.projection && entry.published)
            observer(entry.key[1],pointer(entry),entry.projection->offsets);
        observers_[source].push_back(std::move(observer));
    }
    // Route publication has fenced the previous readers. Check complete hits
    // before acquiring RAM expert slices or reserving transfer-stage slots.
    const void* pin_existing_projection(int64_t source,int64_t expert,const std::vector<int64_t>& sizes,
            std::array<std::size_t,8>& offsets,std::size_t& logical_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=entries_.find({source,expert,-1});if(found==entries_.end())return nullptr;
        auto it=found->second;
        if(!it->projection || sizes.size()!=std::size_t(it->projection->fields))
            throw std::logic_error("KV existing projection field count changed");
        for(std::size_t f=0;f<sizes.size();++f)
            if(sizes[f]<0 || std::size_t(sizes[f])!=it->projection->bytes[f])
                throw std::logic_error("KV existing projection field geometry changed");
        if(it->retired)pending_publications_.push_back(it->key);
        lru_.splice(lru_.begin(),it->retired?retired_:lru_,it);it->retired=false;it->packet=packet_;
        offsets=it->projection->offsets;logical_bytes=it->projection->logical_bytes;hits_+=logical_bytes;
        return pointer(*it);
    }
    const void* pin_projection(int64_t source,int64_t expert,const std::vector<int64_t>& sizes,
            cudaStream_t stream,bool& hit,const ProjectionFill& populate) {
        if(sizes.empty() || sizes.size()>8 || !populate)
            throw std::invalid_argument("invalid KV projection layout");
        ProjectionLayout layout;layout.fields=int(sizes.size());std::size_t total=0;
        for(std::size_t f=0;f<sizes.size();++f) {
            if(sizes[f]<0 || total>std::numeric_limits<std::size_t>::max()-15)
                throw std::overflow_error("KV projection field size overflow");
            total=(total+15)&~std::size_t(15);layout.offsets[f]=total;layout.bytes[f]=std::size_t(sizes[f]);
            if(layout.bytes[f]>std::numeric_limits<std::size_t>::max()-total)
                throw std::overflow_error("KV projection payload overflow");
            total+=layout.bytes[f];layout.logical_bytes+=layout.bytes[f];
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return pin_locked({source,expert,-1},nullptr,total,stream,hit,nullptr,{},&layout,populate);
    }
    void retire_expert(int64_t source,int64_t expert,int fields) {
        std::lock_guard<std::mutex> lock(mutex_);
        for(int f=-1;f<fields;++f)if(auto found=entries_.find({source,expert,f});found!=entries_.end()) {
            auto it=found->second;if(it->retired)continue;
            if(it->published){notify_projection(*it,false);it->published=false;}
            it->retired=true;retired_.splice(retired_.end(),lru_,it);
        }
    }
    // Residency exchange holds the completed window and joins before the next
    // packet/KV growth. It may reuse these bytes until that exchange finishes.
    const void* find_for_exchange(const Key& key,std::size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=entries_.find(key);
        if(found!=entries_.end()) {
            const auto& entry=*found->second;
            if(entry.bytes!=bytes)throw std::logic_error("KV exchange field geometry changed");
            return pointer(entry);
        }
        const auto projection=entries_.find({key[0],key[1],-1});
        if(projection==entries_.end())return nullptr;
        const auto& entry=*projection->second;
        if(!entry.projection || key[2]<0 || key[2]>=entry.projection->fields ||
                entry.projection->bytes[std::size_t(key[2])]!=bytes)
            throw std::logic_error("KV exchange projection field geometry changed");
        return static_cast<const uint8_t*>(pointer(entry))+entry.projection->offsets[std::size_t(key[2])];
    }
    const void* pin(const Key& key,const void* source,std::size_t bytes,cudaStream_t stream,bool& hit,std::vector<DmaCopy>* batch=nullptr,
                    const std::function<const void*()>& stage_source={}) {
        std::lock_guard<std::mutex> lock(mutex_);return pin_locked(key,source,bytes,stream,hit,batch,stage_source);
    }
    bool copy(const Key& key,const void* source,void* destination,std::size_t bytes,cudaStream_t stream,bool& hit) {
        std::lock_guard<std::mutex> lock(mutex_);const auto* cache=pin_locked(key,source,bytes,stream,hit);
        if(cache)MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(destination,cache,bytes,cudaMemcpyDeviceToDevice,stream));
        return cache!=nullptr;
    }
    std::size_t bytes() const {std::lock_guard<std::mutex> lock(mutex_);return bytes_;}
    std::size_t hits() const {std::lock_guard<std::mutex> lock(mutex_);return hits_;}
    std::size_t retired_bytes() const {std::lock_guard<std::mutex> lock(mutex_);return retired_bytes_;}
    double lookup_us() const {std::lock_guard<std::mutex> lock(mutex_);return lookup_us_;}
    std::size_t region_bytes(std::size_t region) const {
        std::lock_guard<std::mutex> lock(mutex_);std::size_t result=0;
        for(const auto* entries:{&lru_,&retired_})for(const auto& e:*entries)if(e.region==region)result+=e.bytes;return result;
    }
private:
    struct Region {mfq_tensor_backend::Tensor owner;int64_t capacity=0,ratio=1,manual_live=-1;std::size_t row_bytes=0,begin=0,end=0,largest_free=0;bool prefix=false;std::map<std::size_t,std::size_t> free;};
    struct Entry {Key key;std::size_t region,first,pages,bytes;uint64_t packet=0;bool retired=false,published=false;
        std::unique_ptr<ProjectionLayout> projection;};
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::size_t hash=0;
            for(const auto value:key)hash^=std::hash<int64_t>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
            return hash;
        }
    };
    const void* pin_locked(const Key& key,const void* source,std::size_t bytes,cudaStream_t stream,bool& hit,std::vector<DmaCopy>* batch=nullptr,
                           const std::function<const void*()>& stage_source={},
                           const ProjectionLayout* projection=nullptr,const ProjectionFill& populate={}) {
        const auto started=std::chrono::steady_clock::now();
        struct Timer {double& total;std::chrono::steady_clock::time_point start;
            ~Timer(){total+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}} timer{lookup_us_,started};
        hit=false;if(bytes<page)return nullptr;
        if(auto found=entries_.find(key);found!=entries_.end()) {
            auto it=found->second;
            if(it->bytes!=bytes)throw std::logic_error("KV expert loan field geometry changed");
            if(projection && (!it->projection || projection->fields!=it->projection->fields ||
                    projection->bytes!=it->projection->bytes))
                throw std::logic_error("KV expert projection geometry changed");
            if(it->retired && it->projection)pending_publications_.push_back(it->key);
            lru_.splice(lru_.begin(),it->retired?retired_:lru_,it);it->retired=false;
            it->packet=packet_;hit=true;hits_+=projection?projection->logical_bytes:bytes;
            return pointer(*it);
        }
        const auto pages=(bytes+page-1)/page;
        if(std::none_of(regions_.begin(),regions_.end(),[&](const Region& r){return r.end>=r.begin && r.end-r.begin>=pages;}))return nullptr;
        auto room=find(pages);
        while(room.first==regions_.size()) {
            auto candidate=lru_.end();
            while(candidate!=lru_.begin()){--candidate;if(candidate->packet!=packet_)break;}
            if(candidate==lru_.end() || candidate->packet==packet_)break;
            erase(candidate);room=find(pages);
        }
        if(room.first==regions_.size())return nullptr;
        // Pageable experts need a pinned upload only on a cache miss. Hits
        // must not copy their weights through CPU staging again.
        if(stage_source)source=stage_source();
        Entry entry{key,room.first,room.second,pages,bytes,packet_};
        if(projection)entry.projection=std::make_unique<ProjectionLayout>(*projection);
        auto* cache=pointer(entry);
        if(!initialization_waited_ || copy_stream_!=stream) {
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream,initialized_.get(),0));
            initialization_waited_=true;
        }
        if(projection)populate(cache,projection->offsets);
        else if(batch)batch->push_back({cache,source,bytes});
        else MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(cache,source,bytes,cudaMemcpyHostToDevice,stream));
        auto& r=regions_[room.first];const auto previous=r.free.at(entry.first),remaining=previous-pages;
        r.free.erase(entry.first);if(remaining)r.free.emplace(entry.first+pages,remaining);
        if(previous==r.largest_free) {
            r.largest_free=0;for(const auto& [begin,size]:r.free)r.largest_free=std::max(r.largest_free,size);
        }
        lru_.push_front(std::move(entry));entries_[key]=lru_.begin();bytes_+=bytes;copy_stream_=stream;
        if(projection)pending_publications_.push_back(key);
        return cache;
    }

    bool update_bounds() {
        bool changed=false;
        for(auto& r:regions_) {
            const auto live=std::min(r.capacity,r.manual_live>=0?r.manual_live:(high_water_+r.ratio-1)/r.ratio);
            const auto begin=r.prefix?0:(std::size_t(live)*r.row_bytes+page-1)/page;
            const auto end=r.prefix?std::size_t(r.capacity-live)*r.row_bytes/page:r.owner.nbytes()/page;
            if(begin==r.begin && end==r.end)continue;
            r.begin=begin;r.end=end;changed=true;
            for(auto it=r.free.begin();it!=r.free.end();) {
                const auto begin=std::max(it->first,r.begin),end=std::min(it->first+it->second,r.end);
                if(begin==it->first && end==it->first+it->second){++it;continue;}
                it=r.free.erase(it);if(end>begin)r.free.emplace(begin,end-begin);
            }
            r.largest_free=0;for(const auto& [begin,size]:r.free)r.largest_free=std::max(r.largest_free,size);
        }
        return changed;
    }
    void revoke() {
        bool revoked=false;
        for(auto* entries:{&lru_,&retired_})for(auto it=entries->begin();it!=entries->end();) {
            const auto& r=regions_[it->region];
            if(it->first<r.begin || it->first+it->pages>r.end){auto old=it++;erase(old);revoked=true;}else ++it;
        }
        if(revoked && copy_stream_) {
            released_.record(copy_stream_);
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(current_stream().stream(),released_.get(),0));
        }
    }
    std::pair<std::size_t,std::size_t> find(std::size_t pages) const {
        for(std::size_t i=0;i<regions_.size();++i)if(regions_[i].largest_free>=pages)
            for(const auto& [begin,size]:regions_[i].free)if(size>=pages)return {i,begin};
        return {regions_.size(),0};
    }
    void* pointer(const Entry& e) const {return static_cast<uint8_t*>(regions_[e.region].owner.data_ptr())+e.first*page;}
    void notify_projection(const Entry& entry,bool valid) {
        if(!entry.projection)return;
        if(auto found=observers_.find(entry.key[0]);found!=observers_.end())
            for(const auto& observer:found->second)
                observer(entry.key[1],valid?pointer(entry):nullptr,entry.projection->offsets);
    }
    void erase(std::list<Entry>::iterator it) {
        if(it->published)notify_projection(*it,false);
        auto& r=regions_[it->region];
        auto begin=std::max(it->first,r.begin),end=std::min(it->first+it->pages,r.end);
        if(begin<end) {
            auto next=r.free.lower_bound(begin);
            if(next!=r.free.begin()) {auto prior=std::prev(next);if(prior->first+prior->second==begin){begin=prior->first;r.free.erase(prior);}}
            if(next!=r.free.end() && next->first==end){end=next->first+next->second;r.free.erase(next);}
            r.free.emplace(begin,end-begin);
            r.largest_free=std::max(r.largest_free,end-begin);
        }
        bytes_-=it->bytes;entries_.erase(it->key);
        (it->retired?retired_:lru_).erase(it);
    }
    mutable std::mutex mutex_;
    std::vector<Region> regions_;
    std::list<Entry> lru_,retired_;
    std::unordered_map<Key,std::list<Entry>::iterator,KeyHash> entries_;
    std::unordered_map<int64_t,std::vector<ProjectionObserver>> observers_;
    std::vector<Key> pending_publications_;
    uint64_t packet_=1;
    std::size_t bytes_=0,hits_=0,retired_bytes_=0;
    double lookup_us_=0;
    int64_t high_water_=0;
    cudaStream_t copy_stream_=nullptr;
    bool initialization_waited_=false;
    Event initialized_,released_;
};
}
