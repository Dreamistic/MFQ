#include "runtime/moe_host_expert_cache.h"

#include <chrono>
#include <array>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace tb=mfq_tensor_backend;
namespace {
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
MixedMoePool weight(std::size_t bytes,int marker) {
    MixedMoePool result;
    result.local_experts=1;
    result.nint.q_packed=tb::full({1,static_cast<std::int64_t>(bytes)},marker,
        tb::TensorOptions().dtype(tb::kUInt8));
    return result;
}
mfq::MoeCacheKey key(int expert,int source=0) { return {source,0,expert}; }
void bytes_equal(MoeHostExpertCache::Lease value,int marker) {
    check(value && value->weights.nint.q_packed.numel()==value->bytes,"host lease shape mismatch");
    const auto* data=value->weights.nint.q_packed.data_ptr<std::uint8_t>();
    for (std::size_t i=0; i<value->bytes; ++i) check(data[i]==marker,"host lease payload changed");
}
void wait_coalesced(const MoeHostExpertCache& cache,std::promise<void>& release) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while (!cache.stats().coalesced && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
    release.set_value();
    check(cache.stats().coalesced==1,"same-key host read did not coalesce");
}
}

int main() try {
    for(bool reuse:{false,true}) {
        MoeHostExpertCache cache(192);if(reuse)cache.enable_reuse_priority();
        auto hints=std::make_shared<MoeHostExpertCache::DeviceCacheHints>(
            std::vector<mfq::MoeCacheKey>{key(0),key(1),key(2),key(3),key(4)});
        cache.set_device_cache_hints(hints);
        for(int e:{0,1,2})cache.acquire(key(e),64,[e]{return weight(64,e+1);});
        bytes_equal(cache.find(key(1)),2);
        hints->publish(1,true);
        cache.acquire(key(3),64,[]{return weight(64,4);});
        check(cache.contains(key(0)) && !cache.contains(key(1)) && cache.contains(key(2)),
            "GPU-duplicated RAM image displaced a unique RAM expert");
        check(cache.stats().device_duplicate_evictions==1 && cache.stats().device_duplicate_evicted_bytes==64,
            "duplicate RAM eviction accounting differs from released payloads");
        // A late removal supersedes an earlier publication of the same key.
        hints->publish(3,true);hints->publish(3,false);
        cache.acquire(key(4),64,[]{return weight(64,5);});
        check(cache.contains(key(3)) && !cache.contains(key(0)),"revoked GPU hint still biased RAM eviction");
        check(cache.stats().managed_peak_bytes==192,"exclusive cache exceeded its byte budget");
    }
    {
        MoeHostExpertCache cache(128);
        auto hints=std::make_shared<MoeHostExpertCache::DeviceCacheHints>(std::vector<mfq::MoeCacheKey>{key(1),key(2)});
        cache.set_device_cache_hints(hints);
        cache.acquire(key(0),64,[]{return weight(64,1);});
        auto held=cache.acquire(key(1),64,[]{return weight(64,2);});
        hints->publish(0,true);
        cache.acquire(key(2),64,[]{return weight(64,3);});
        check(cache.contains(key(1)) && !cache.contains(key(0)),"GPU hint evicted a leased RAM image");
        bytes_equal(held,2);held.reset();
        hints->publish(1,true);
        cache.set_device_cache_hints({});
        cache.find(key(1));
        cache.acquire(key(3),64,[]{return weight(64,4);});
        check(cache.contains(key(1)) && !cache.contains(key(2)),"detached GPU cache retained eviction priority");
    }
    {
        MoeHostExpertCache cache(128);cache.preserve_resident_experts();
        auto hints=std::make_shared<MoeHostExpertCache::DeviceCacheHints>(std::vector<mfq::MoeCacheKey>{key(0)});
        cache.set_device_cache_hints(hints);
        cache.acquire(key(0),64,[]{return weight(64,1);});hints->publish(0,true);
        cache.acquire(key(1),64,[]{return weight(64,2);});
        check(cache.contains(key(0)) && !cache.stats().evictions,"GPU hint removed an image without RAM pressure");
        cache.acquire(key(2),64,[]{return weight(64,3);});
        check(cache.contains(key(0)) && cache.contains(key(1)) && !cache.contains(key(2)),
            "GPU hint discarded the last reserved RAM copy in complete residency mode");
    }
    for(bool batch:{false,true}) {
        MoeHostExpertCache cache(128);
        auto hints=std::make_shared<MoeHostExpertCache::DeviceCacheHints>(std::vector<mfq::MoeCacheKey>{key(0),key(2)});
        cache.set_device_cache_hints(hints);hints->publish(0,true);
        cache.acquire(key(0),64,[]{return weight(64,1);});
        cache.acquire(key(1),64,[]{return weight(64,2);});
        auto held=cache.find(key(0));
        if(batch)check(cache.exchange_batch({{key(0),key(2),&held,
                [](const MixedMoePool&){return weight(64,3);},[](const MixedMoePool&){}}}),"duplicate batch exchange failed");
        else check(cache.exchange(key(0),key(2),held,[](const MixedMoePool&){return weight(64,3);}),"duplicate exchange failed");
        held.reset();hints->publish(1,true);
        cache.acquire(key(3),64,[]{return weight(64,4);});
        check(!cache.contains(key(2)) && cache.contains(key(1)),"exchange corrupted the duplicate eviction queue");
        check(cache.stats().device_duplicate_evictions==1,"exchanged duplicate was omitted from eviction accounting");
    }
    {
        MoeHostExpertCache cache(192);cache.enable_reuse_priority();
        bytes_equal(cache.acquire(key(0),64,[]{return weight(64,1);}),1);
        bytes_equal(cache.find(key(0)),1);
        bytes_equal(cache.acquire_scan(key(1),64,[]{return weight(64,2);}),2);
        bytes_equal(cache.acquire_scan(key(1),64,[]{throw std::runtime_error("scan hit reread");return weight(64,0);}),2);
        check(cache.stats().reuse_promotions==1,"scan acquisition promoted one-use data");
        std::promise<void> entered,release;auto released=release.get_future().share();
        auto loading=std::async(std::launch::async,[&] {
            return cache.acquire_scan(key(2),64,[&]{entered.set_value();released.wait();return weight(64,3);});
        });
        entered.get_future().wait();
        int unexpected=0;
        const auto rejected=cache.acquire_scan(key(3),64,[&]{++unexpected;return weight(64,4);});
        release.set_value();bytes_equal(loading.get(),3);
        check(!rejected && !unexpected,"a scan evicted entries or ignored an in-flight reservation");
        check(cache.contains(key(0)) && cache.contains(key(1)) && cache.contains(key(2)),"scan displaced cached experts");
        check(cache.stats().managed_peak_bytes==192 && !cache.stats().evictions &&
            cache.stats().scan_admissions==2 && cache.stats().scan_admitted_bytes==128,
            "scan admission accounting exceeded the RAM budget");
        cache.erase(key(2));
        bool failed=false;
        try{cache.acquire_scan(key(3),64,[]{throw std::runtime_error("expected scan read failure");return weight(64,0);});}
        catch(const std::runtime_error&){failed=true;}
        check(failed && cache.stats().managed_bytes==128,"failed scan retained a reservation");
        bytes_equal(cache.acquire_scan(key(3),64,[]{return weight(64,4);}),4);
    }
    for(bool reuse:{false,true}) {
        MoeHostExpertCache cache(512);if(reuse)cache.enable_reuse_priority();
        for(int expert:{0,1})cache.acquire(key(expert),64,[expert]{return weight(64,expert+1);});
        if(reuse) {
            cache.set_reuse_learning(false);
            for(int expert:{0,1})bytes_equal(cache.find(key(expert)),expert+1);
            check(!cache.stats().reuse_promotions,"synthetic accesses promoted RAM cache entries");
            cache.set_reuse_learning(true);
        }
        for(int expert:{0,1})bytes_equal(cache.find(key(expert)),expert+1);
        cache.acquire(key(99),64,[]{return weight(64,3);});
        for(int scan=0;scan<3;++scan)bytes_equal(cache.find(key(99),false),3);
        for(int expert=2;expert<34;++expert)cache.acquire(key(expert),64,[expert]{return weight(64,expert+1);});
        check(cache.contains(key(0))==reuse && cache.contains(key(1))==reuse,
            "a one-use RAM scan displaced repeated decoded expert images");
        if(!reuse)continue;
        check(cache.stats().reuse_promotions==2,"prefill image lookups promoted one-use RAM experts");
        for(int expert=34;expert<50;++expert) {
            cache.acquire(key(expert),64,[expert]{return weight(64,expert+1);});
            bytes_equal(cache.find(key(expert)),expert+1);
            check(cache.stats().reuse_protected_bytes<=384,"RAM reuse protection exceeded its byte quota");
        }
        cache.acquire(key(0,-2),64,[]{return weight(64,9);});
        bytes_equal(cache.find(key(0,-2)),9);
        check(cache.reclaim_optional(64)==64,"repeated optional backup was protected from reclamation");
        auto held=cache.find(key(49));check(bool(held),"RAM reuse fixture lost its latest expert");
        const auto protected_before=cache.stats().reuse_protected_bytes;
        check(cache.exchange(key(49),key(50),held,[](const MixedMoePool&){return weight(64,51);}),
            "RAM reuse prevented a valid GPU exchange");
        check(cache.stats().reuse_protected_bytes+64==protected_before,"GPU victim inherited the incoming expert's protection");
        held.reset();bytes_equal(cache.find(key(50)),51);
        check(cache.stats().managed_peak_bytes<=512,"RAM reuse exceeded its physical payload budget");
    }
    {
        MoeHostExpertCache cache(0);
        int reads=0;
        const auto load=[&]{++reads;return weight(64,23);};
        auto first=cache.acquire(key(3),64,load);
        auto second=cache.acquire(key(3),64,load);
        check(first==second && reads==1,"active bypassed expert was reread for another token/projection");
        check(cache.stats().managed_bytes==0 && cache.stats().transient_bytes==64,"bypassed expert accounting duplicated a lease");
        first.reset();bytes_equal(second,23);second.reset();
        check(cache.stats().transient_bytes==0,"expired bypassed expert retained its RAM payload");
        auto next=cache.acquire(key(3),64,load);
        check(reads==2,"expired transient weak lease returned stale weights");bytes_equal(next,23);
    }
    {
        MoeHostExpertCache cache(256);
        check(!cache.find(key(9,-2)),"missing immutable image created a lease");
        cache.acquire(key(0,-2),128,[]{return weight(128,17);});
        auto held=cache.find(key(0,-2));bytes_equal(held,17);
        check(cache.reclaim_optional(128)==0,"active immutable image was reclaimed");
        cache.erase(key(0,-2));bytes_equal(held,17);
        check(cache.stats().managed_bytes==128,"erased active image lost its reservation");
        held.reset();check(cache.stats().managed_bytes==0,"immutable image lease leaked");
    }
    {
        MoeHostExpertCache cache(1000);
        cache.acquire(key(0),400,[] { return weight(400,1); });
        cache.acquire(key(1),300,[] { return weight(300,2); });
        cache.acquire(key(2),300,[] { return weight(300,3); });
        auto pinned=cache.acquire(key(0),400,[] { throw std::runtime_error("unexpected reread"); return weight(400,0); });
        cache.acquire(key(3),500,[] { return weight(500,4); });
        check(cache.contains(key(0)) && cache.contains(key(3)) &&
            !cache.contains(key(1)) && !cache.contains(key(2)),"byte LRU evicted the wrong entries");
        bytes_equal(pinned,1);
        cache.promote(key(0));
        check(cache.stats().managed_bytes==900,"promoted pinned lease was no longer charged");
        auto temporary=cache.acquire(key(4),800,[] { return weight(800,5); });
        check(!cache.contains(key(4)) && cache.stats().managed_bytes==400 &&
            cache.stats().transient_bytes==800,"pinned budget pressure exceeded managed capacity");
        pinned.reset(); temporary.reset();
        check(!cache.stats().managed_bytes && !cache.stats().transient_bytes,"host leases did not release their reservations");
        check(cache.stats().managed_peak_bytes<=1000,"managed RAM exceeded byte budget");
    }
    {
        MoeHostExpertCache cache(512);
        int copies=0;
        check(cache.demote(key(0),512,[&] { ++copies; return weight(512,9); }),"GPU victim was not admitted");
        auto lease=cache.acquire(key(0),512,[] { throw std::runtime_error("demoted weight reread SSD"); return weight(512,0); });
        bytes_equal(lease,9);
        cache.promote(key(0));
        check(!cache.contains(key(0)) && cache.stats().managed_bytes==512,"GPU promotion retained a persistent RAM duplicate");
        check(!cache.demote(key(1),512,[&] { ++copies; return weight(512,1); }) && copies==1,
            "pinned budget overflow performed an unnecessary GPU readback");
        lease.reset();
        check(cache.demote(key(1),512,[&] { ++copies; return weight(512,1); }),"released capacity was not reused");
        auto distinct=cache.acquire(key(1,1),512,[] { return weight(512,2); });
        bytes_equal(distinct,2);
        check(!cache.contains(key(1)) && cache.contains(key(1,1)),"source IDs shared a host cache entry");
    }
    {
        for (std::size_t budget:{0,128}) {
            MoeHostExpertCache cache(budget);
            auto lease=cache.acquire(key(0),512,[] { return weight(512,6); });
            check(!cache.stats().resident_entries && !cache.stats().managed_bytes &&
                cache.stats().transient_bytes==512,"oversized/disabled cache did not use bounded-route scratch");
            lease.reset();
            check(!cache.stats().transient_bytes,"uncached scratch reservation leaked");
        }
    }
    for (bool fail:{false,true}) {
        MoeHostExpertCache cache(64);
        std::promise<void> started,release;
        auto gate=release.get_future().share();
        int reads=0;
        auto first=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[&] {
                ++reads; started.set_value(); gate.wait();
                if (fail) throw std::runtime_error("injected host source failure");
                return weight(64,7);
            });
        });
        started.get_future().wait();
        auto second=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[] { throw std::runtime_error("duplicate host source read"); return weight(64,0); });
        });
        wait_coalesced(cache,release);
        if (fail) {
            for (auto* future:{&first,&second}) {
                bool rejected=false;
                try { future->get(); } catch (const std::runtime_error& error) {
                    rejected=std::string(error.what())=="injected host source failure";
                }
                check(rejected,"coalesced read failure was not propagated");
            }
            check(!cache.stats().managed_bytes && !cache.stats().resident_entries,"failed read left a reservation or resident entry");
            bytes_equal(cache.acquire(key(0),64,[] { return weight(64,8); }),8);
        } else {
            auto a=first.get(),b=second.get();
            check(a.get()==b.get() && reads==1,"coalesced readers did not share immutable data");
            bytes_equal(a,7);
        }
    }
    {
        MoeHostExpertCache cache(64);
        std::promise<void> started,release;
        auto gate=release.get_future().share();
        auto pending=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[&] { started.set_value(); gate.wait(); return weight(64,3); });
        });
        started.get_future().wait();
        cache.promote(key(0));
        release.set_value();
        auto lease=pending.get();
        check(!cache.contains(key(0)) && cache.stats().managed_bytes==64,"inflight promotion published a stale RAM duplicate");
        lease.reset();
        check(!cache.stats().managed_bytes,"inflight promotion lost its reservation owner");
    }
    {
        MoeHostExpertCache::Lease retained;
        {
            MoeHostExpertCache cache(64);
            retained=cache.acquire(key(0),64,[] { return weight(64,4); });
        }
        bytes_equal(retained,4);
    }
    std::cout<<"host expert byte LRU, leases, promotion/demotion, concurrent coalescing and failed-read recovery passed\n";
    {
        MoeHostExpertCache cache(128);
        cache.acquire(key(0),64,[] { return weight(64,1); });
        cache.acquire(key(1),64,[] { return weight(64,2); });
        cache.preserve_resident_experts();
        auto scratch=cache.acquire(key(2),64,[] { return weight(64,3); });
        check(cache.contains(key(0)) && cache.contains(key(1)) && !cache.contains(key(2)) &&
            cache.stats().evictions==0,"complete residency evicted the last RAM expert copy");
        scratch.reset();
        cache.promote(key(0));
        check(cache.demote(key(2),64,[] { return weight(64,3); }),"complete residency did not reuse promoted capacity");
    }
    {
        MoeHostExpertCache cache(64);cache.preserve_resident_experts();
        auto incoming=cache.acquire(key(0),64,[]{return weight(64,1);});
        auto reader=incoming;int copies=0;
        auto exchange=[&](const MixedMoePool& source){
            ++copies;auto value=source;
            std::memset(value.nint.q_packed.data_ptr(),9,64);return value;
        };
        check(!cache.exchange(key(0),key(1),incoming,exchange) && copies==0,"tier exchange overwrote a live CPU lease");
        reader.reset();const auto* before=incoming->weights.nint.q_packed.data_ptr();
        bool failed=false;
        try{cache.exchange(key(0),key(1),incoming,[](const auto&)->MixedMoePool{throw std::runtime_error("copy failed");});}
        catch(const std::runtime_error&){failed=true;}
        check(failed && cache.contains(key(0)) && !cache.contains(key(1)),"failed tier exchange changed RAM ownership");
        check(cache.exchange(key(0),key(1),incoming,exchange),"tier exchange failed without other readers");
        incoming.reset();auto outgoing=cache.acquire(key(1),64,[]{throw std::runtime_error("exchange reread SSD");return weight(64,0);});
        bytes_equal(outgoing,9);
        check(outgoing->weights.nint.q_packed.data_ptr()==before && cache.stats().managed_bytes==64 &&
            cache.stats().managed_peak_bytes==64 && !cache.contains(key(0)),"tier exchange grew or copied the RAM allocation");
    }
    {
        MoeHostExpertCache cache(192);cache.preserve_resident_experts();
        cache.acquire(key(0),128,[]{return weight(128,1);});
        auto shadow=cache.acquire(key(0,-2),64,[]{return weight(64,7);});
        int copies=0;
        check(!cache.demote(key(1),64,[&]{++copies;return weight(64,2);}) && copies==0,
            "active prefill backup lost its only CPU copy");
        shadow.reset();
        check(cache.demote(key(1),64,[&]{++copies;return weight(64,2);}) && copies==1,
            "optional prefill backup blocked an expert demotion");
        check(cache.contains(key(0)) && cache.contains(key(1)) && !cache.contains(key(0,-2)) &&
            cache.stats().managed_peak_bytes<=192,"prefill backup evicted a last expert copy or exceeded RAM budget");
    }
    {
        MoeHostExpertCache cache(512);cache.preserve_resident_experts();
        cache.acquire(key(0),192,[]{return weight(192,1);});
        cache.acquire(key(0,-2),64,[]{return weight(64,2);});
        cache.acquire(key(1,-2),96,[]{return weight(96,3);});
        auto active=cache.acquire(key(2,-2),32,[]{return weight(32,4);});
        cache.acquire(key(0,-2),64,[]{throw std::runtime_error("backup reread");return weight(64,0);});
        check(cache.stats().optional_bytes==192,"backup accounting includes required experts");
        check(cache.reclaim_optional(0)==0 && cache.reclaim_optional(1)==96,
            "pressure reclaim ignored backup LRU order");
        check(!cache.contains(key(1,-2)) && cache.contains(key(0,-2)) && cache.contains(key(0)),
            "pressure reclaim released a required expert or the newest backup");
        check(cache.reclaim_optional(512)==64 && cache.stats().optional_bytes==32,
            "pressure reclaim freed an active backup lease");
        bytes_equal(active,4);active.reset();
        check(cache.reclaim_optional(512)==32 && cache.stats().resident_bytes==192 &&
            cache.stats().optional_bytes==0 && cache.stats().optional_reclaimed_bytes==192,
            "pressure reclaim lost bytes or the only ordinary expert copy");
    }
    {
        MoeHostExpertCache cache(128);
        std::promise<void> started,release;auto gate=release.get_future().share();
        auto pending=std::async(std::launch::async,[&] {
            return cache.acquire(key(0,-2),64,[&] {started.set_value();gate.wait();return weight(64,9);});
        });
        started.get_future().wait();
        const auto released=cache.reclaim_optional(128);
        release.set_value();auto active=pending.get();
        check(released==0,"pressure reclaim released an unpublished backup");
        check(cache.reclaim_optional(128)==0,"pressure reclaim released the completed loader's lease");
        bytes_equal(active,9);active.reset();
        check(cache.reclaim_optional(128)==64 && cache.stats().managed_bytes==0,
            "pressure reclaim could not reuse the completed backup allocation");
    }
    int batch_cases=0;
    for(bool reuse:{false,true})for(bool bulk:{false,true})for(int fail_at:{-2,0,1,2,3,-1}) {
        if(!bulk && fail_at==-2)continue;
        MoeHostExpertCache cache(192);cache.preserve_resident_experts();
        if(reuse)cache.enable_reuse_priority();
        std::array<MoeHostExpertCache::Lease,3> held;
        for(int i=0;i<3;++i)held[i]=cache.acquire(key(i),64,[i]{return weight(64,i+1);});
        if(reuse)for(int i=0;i<3;++i)bytes_equal(cache.find(key(i)),i+1);
        std::vector<MoeHostExpertCache::Exchange> changes;
        for(int i=0;i<3;++i)changes.push_back({key(i),key(i+3),&held[i],
            [i,fail_at,bulk](const MixedMoePool& original) {
                auto result=original;if(!bulk)std::memset(result.nint.q_packed.data_ptr(),i+7,64);
                if(fail_at==i)throw std::runtime_error("partial RAM reuse failure");return result;
            },[i](const MixedMoePool& original){std::memset(original.nint.q_packed.data_ptr(),i+1,64);}});
        int copied=0;
        std::function<void()> prepare;
        if(bulk)prepare=[&] {
            ++copied;
            for(int i=0;i<3;++i) {
                std::memset(held[i]->weights.nint.q_packed.data_ptr(),i+7,64);
                if(fail_at==-2 && i==1)throw std::runtime_error("partial bulk RAM reuse failure");
            }
        };
        auto reader=held[1];
        check(!cache.exchange_batch(changes,[&]{++copied;},prepare),"batch reused a CPU reader's fields");
        check(!copied,"rejected batch published a GPU mapping");reader.reset();
        bool failed=false;
        try {check(cache.exchange_batch(changes,[&]{if(fail_at==3)throw std::runtime_error("publication failure");},prepare),"batch was rejected without readers");}
        catch(const std::runtime_error&){failed=true;}
        check(failed==(fail_at>=0 || fail_at==-2),"batch failure did not propagate");
        for(int i=0;i<3;++i) {
            if(failed){check(cache.contains(key(i)) && !cache.contains(key(i+3)),"batch rollback changed an entry");bytes_equal(held[i],i+1);}
            else {
                held[i].reset();auto value=cache.acquire(key(i+3),64,[]{throw std::runtime_error("batch reread SSD");return weight(64,0);});
                bytes_equal(value,i+7);check(!cache.contains(key(i)),"batch retained the promoted key");
            }
        }
        check(cache.stats().managed_bytes==192 && cache.stats().managed_peak_bytes==192,"batch duplicated RAM payloads");
        ++batch_cases;
    }
    int keyed_cases=0;
    if(const auto* option=std::getenv("MFQ_MOE_KEYED_EXCHANGE");option && option[0]=='1')
      for(bool reuse:{false,true})for(bool fail:{false,true}) {
        MoeHostExpertCache cache(192);cache.preserve_resident_experts();
        if(reuse)cache.enable_reuse_priority();
        auto held=cache.acquire(key(0),64,[]{return weight(64,1);});
        cache.acquire(key(7),64,[]{return weight(64,7);});
        if(reuse)bytes_equal(cache.find(key(0)),1);
        std::promise<void> entered,release;
        auto released=release.get_future().share();
        std::vector<MoeHostExpertCache::Exchange> changes={{key(0),key(1),&held,
            [](const MixedMoePool& original){return original;},
            [](const MixedMoePool& original){std::memset(original.nint.q_packed.data_ptr(),1,64);}}};
        auto exchange=std::async(std::launch::async,[&] {
            try {
                const bool done=cache.exchange_batch(changes,[&] {
                    if(fail)throw std::runtime_error("expected keyed publication failure");
                },[&] {
                    std::memset(held->weights.nint.q_packed.data_ptr(),9,64);
                    entered.set_value();released.wait();
                });
                return done && !fail;
            }catch(const std::runtime_error&){return fail;}
        });
        entered.get_future().wait();
        auto preflight=std::async(std::launch::async,[&] {
            return !cache.try_find(key(0)) && !cache.try_find(key(1)) && bool(cache.try_find(key(7),false));
        });
        const bool preflight_ready=preflight.wait_for(std::chrono::seconds(1))==std::future_status::ready;
        auto unrelated=std::async(std::launch::async,[&] {
            bytes_equal(cache.acquire(key(7),64,[]{throw std::runtime_error("unrelated reread");return weight(64,0);}),7);
            bytes_equal(cache.acquire(key(8),64,[]{return weight(64,8);}),8);
            if(reuse)bytes_equal(cache.find(key(8)),8);
            const auto stats=cache.stats();
            return stats.resident_entries==3 && stats.managed_bytes==192 && stats.managed_peak_bytes==192;
        });
        auto old_reader=std::async(std::launch::async,[&]{return cache.find(key(0));});
        auto new_reader=std::async(std::launch::async,[&]{return cache.find(key(1));});
        const bool unrelated_ready=unrelated.wait_for(std::chrono::seconds(1))==std::future_status::ready;
        const bool old_blocked=old_reader.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
        const bool new_blocked=new_reader.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
        // Release all workers before reporting a failure, including a legacy
        // global-lock regression; future destruction must never deadlock.
        release.set_value();
        check(exchange.get(),"keyed publication/rollback failed");
        check(preflight_ready && preflight.get(),"RAM preflight waited for a key publication or read partially exchanged fields");
        check(unrelated_ready && unrelated.get(),"unrelated host expert was blocked or accounting changed");
        auto old=old_reader.get(),next=new_reader.get();
        check(old_blocked && new_blocked,"reader saw partially exchanged RAM");
        if(fail){bytes_equal(old,1);check(!next,"failed keyed exchange published outgoing payload");}
        else {check(!old,"keyed exchange retained incoming payload");bytes_equal(next,9);}
        ++keyed_cases;
      }
    std::cout<<"atomic RAM exchange rollback, reader exclusion and publication checks passed batch_cases="<<batch_cases
        <<" keyed_concurrency_cases="<<keyed_cases<<"\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
