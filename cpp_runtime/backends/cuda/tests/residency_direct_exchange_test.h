#pragma once

#include "storage/moe_cached_source_internal.h"

// Seed real compressed transfer-cache fields without launching the unrelated
// FFN consumer graph; this fixture can also run under Compute Sanitizer.
struct MoeResidencyTestAccess {
    static void seed_exchange_wire(MoeExpertCache& cache,int64_t bytes) {
        cache.pipeline_wire_gpu_=tb::full({bytes+4096},91,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void check_exchange_wire(const MoeExpertCache& cache) {
        const auto& wire=cache.pipeline_wire_gpu_;
        if(!wire.defined())return;
        for(const auto offset:{int64_t(0),wire.numel()-256}) {
            std::array<uint8_t,256> bytes;
            MFQ_CUDA_CHECK(cudaMemcpy(bytes.data(),wire.data_ptr<uint8_t>()+offset,bytes.size(),cudaMemcpyDeviceToHost));
            if(std::any_of(bytes.begin(),bytes.end(),[](auto v){return v!=91;}))
                throw std::runtime_error("residency exchange overwrote wire guards");
        }
    }
    static bool wire_payload_used(const MoeExpertCache& cache) {
        std::array<uint8_t,256> payload;
        MFQ_CUDA_CHECK(cudaMemcpy(payload.data(),cache.pipeline_wire_gpu_.data_ptr<uint8_t>()+256,
            payload.size(),cudaMemcpyDeviceToHost));
        return std::any_of(payload.begin(),payload.end(),[](auto v){return v!=91;});
    }
    static bool transfer_enabled(const MoeExpertCache& cache){return cache.pipeline_transfer_cache_;}
    static void verify_transfer_fields(MoeExpertCache& cache) {
        for(const auto& source:cache.sources_)for(const auto& cohort:source->cohorts_) {
            const auto key=source->projection_role_+":"+cohort.arena->signature;
            auto found=cache.pipeline_stages_.find(key);
            if(found==cache.pipeline_stages_.end() || !found->second->book)continue;
            const auto& arena=*found->second;
            for(int e=0;e<source->n_experts();++e) {
                const int slot=arena.book->slot_for({source->id_,cohort.index,e});if(slot<0)continue;
                const auto held=source->quant_source_->acquire_expert(e);
                const auto fields=moe_cache_fields(held->weights);
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                    std::vector<uint8_t> actual(static_cast<std::size_t>(bytes));
                    MFQ_CUDA_CHECK(cudaMemcpy(actual.data(),static_cast<const uint8_t*>(arena.fields[f].data_ptr())+slot*bytes,
                        actual.size(),cudaMemcpyDeviceToHost));
                    if(std::memcmp(actual.data(),fields[f].data_ptr(),actual.size()))
                        throw std::runtime_error("transfer field bytes differ: source="+std::to_string(source->id_)+
                            " expert="+std::to_string(e)+" field="+std::to_string(f));
                }
            }
        }
    }
    static void verify_defaults_guard(MoeExpertCache& cache) {
        const auto saved=cache.ram_pcie_fraction_;
        cache.ram_pcie_fraction_=0.5;
        if(MoeResidencyManager(&cache).stats().exchange_defaults)throw std::runtime_error("mixed CPU routing enabled exchange defaults");
        cache.ram_pcie_fraction_=saved;
        cache.pipeline_shared_cpu_cost_=true;
        if(MoeResidencyManager(&cache).stats().exchange_defaults)throw std::runtime_error("CPU calibration enabled exchange defaults");
        cache.pipeline_shared_cpu_cost_=false;
    }
    static void seed(MoeExpertCache& cache,int projection,int expert) {
        auto& source=*cache.sources_.at(projection);
        const int c=source.expert_to_cohort_.at(expert);auto& cohort=source.cohorts_.at(c);
        const auto key=std::to_string(projection)+":"+cohort.arena->signature;
        auto& arena=cache.pipeline_stages_[key];
        if(!arena) {
            arena=std::make_unique<MoeGpuArena>();arena->signature=key;
            arena->slot_bytes=cohort.arena->slot_bytes;arena->slots=1;arena->layouts=cohort.arena->layouts;
            arena->book=std::make_unique<mfq::MoeCacheSlotBook>(1);
        }
        const auto held=source.quant_source_->acquire_expert(expert);
        const auto fields=moe_cache_fields(held->weights);
        if(arena->fields.empty())for(const auto& field:fields)arena->fields.push_back(field.to(tb::kCUDA));
        else for(std::size_t f=0;f<fields.size();++f)arena->fields[f].copy_(fields[f].to(tb::kCUDA));
        arena->book->acquire(source.quant_source_->host_key(expert));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    static void compare_gpu(MoeExpertCache& cache,int projection,int expert,const MixedMoePool& original) {
        auto& source=*cache.sources_.at(projection);
        const int c=source.expert_to_cohort_.at(expert);const auto& cohort=source.cohorts_.at(c);
        const int slot=cohort.host_map.at(expert);
        if(slot<0 || slot!=cohort.arena->book->slot_for(source.quant_source_->host_key(expert)))
            throw std::runtime_error("direct exchange GPU map/book ownership mismatch");
        std::vector<int32_t> device_map(cohort.host_map.size());
        MFQ_CUDA_CHECK(cudaMemcpy(device_map.data(),cohort.active.expert_local.data_ptr<int32_t>(),
            device_map.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));
        if(device_map!=cohort.host_map)throw std::runtime_error("direct exchange published a stale device map");
        const auto fields=moe_cache_fields(original);
        for(std::size_t f=0;f<fields.size();++f) {
            const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
            std::vector<uint8_t> actual(static_cast<std::size_t>(bytes));
            const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(slot)*bytes;
            MFQ_CUDA_CHECK(cudaMemcpy(actual.data(),gpu,actual.size(),cudaMemcpyDeviceToHost));
            if(std::memcmp(actual.data(),fields[f].data_ptr(),actual.size()))
                throw std::runtime_error("direct exchange changed a GPU weight byte");
        }
    }
};

void verify_direct_exchange(const std::filesystem::path& root) {
    int cases=0,repeats=0;
    const char* batching_env=std::getenv("MFQ_MOE_RESIDENCY_CACHED_REFILL");
    const bool defaults=std::getenv("MFQ_TEST_RESIDENCY_EXPECT_DEFAULTS")!=nullptr;
    const bool batching=defaults || (batching_env && std::string(batching_env)=="1");
    for(const std::string format:{"nint5","nvq3j-512"})
    for(const int cached_projections:{0,1,3})
    for(const bool gpu_wire:{false,true})
    for(const std::string phase:{"none","direct_backup","direct_backup_batch","direct_refill","direct_refill_batch","upload","map","host","publish","reader","window","cancel","prefill"}) {
        if(phase=="direct_refill_batch" && (!batching || (!cached_projections && !gpu_wire)))continue;
        const auto* small_backup=std::getenv("MFQ_MOE_RESIDENCY_SMALL_BACKUP_BYTES");
        if(phase=="direct_backup_batch" && small_backup && std::atoll(small_backup)<=0)continue;
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=true;execution.config.moe_ram_pcie_fraction=1.0;
        execution.config.moe_direct_ram=true;execution.config.moe_two_stage_ffn=true;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
        std::array<std::shared_ptr<MfeWeight>,3> weights;int64_t budget=0;
        for(int p=0;p<3;++p) {
            sources[p]=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
            budget+=bytes(sources[p]->metadata()->pools[0]);
        }
        execution.config.moe_host_cache_bytes=3*budget;
        auto cache=make_moe_expert_cache(budget,execution.config);
        for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(
            cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
        auto forward=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
        finalize_moe_expert_cache(cache);
        if(gpu_wire) {
            int64_t incoming_bytes=budget;
            for(int p=0;p<cached_projections;++p)incoming_bytes-=bytes(sources[p]->metadata()->pools[0]);
            MoeResidencyTestAccess::seed_exchange_wire(*cache,incoming_bytes);
        }
        if(!MoeResidencyTestAccess::transfer_enabled(*cache))throw std::runtime_error("direct fixture needs transfer caching");
        for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,1);
        if(defaults)MoeResidencyTestAccess::verify_defaults_guard(*cache);
        MoeHostExpertCache::Lease reader;std::atomic_bool complete=false;
        auto manager=std::make_unique<MoeResidencyManager>(cache.get(),[&](const char* event,std::size_t index) {
            if(std::string(event)=="copies_complete")complete=true;
            if(index)return;
            if(phase==event)throw std::runtime_error("injected direct exchange failure");
            if(phase=="reader" && std::string(event)=="before_direct_exchange")reader=sources[0]->acquire_expert(1);
        });
        if(defaults && !manager->stats().exchange_defaults)throw std::runtime_error("eligible fixture did not enable exchange defaults");
        const bool window=phase=="window" || phase=="cancel";
        if(window) {
            manager->before_layer(0);
            manager->record_window_expert_fence(0,mfq_current_cuda_stream());
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        if(phase=="prefill") {
            auto old_routes=tb::full({32,1},2,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,old_routes.data_ptr<int32_t>(),32,1,mfq_current_cuda_stream());
            auto wide_routes=tb::full({32,2},2,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,wide_routes.data_ptr<int32_t>(),32,2,mfq_current_cuda_stream());
            auto routes=tb::full({32,1},1,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
            manager->record_prefill_routes(0,routes.data_ptr<int32_t>(),32,1,mfq_current_cuda_stream());
            if(manager->stats().committed_projections)throw std::runtime_error("prefill routes exchanged before decode boundary");
            manager->prepare_decode();
            manager->prepare_decode(); // No second transaction without a new prompt.
        } else manager->after_layer(0,std::vector<int32_t>(35,1),35,window);
        if(window) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(!complete.load()) {
                if(std::chrono::steady_clock::now()>deadline) {
                    manager->finish_window(0);throw std::runtime_error("direct exchange did not reach its publication barrier");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(manager->stats().committed_projections)throw std::runtime_error("direct exchange published before window finish");
            for(const auto& source:sources)if(!source->gpu_resident(0) || source->gpu_resident(1))
                throw std::runtime_error("direct exchange changed early tier ownership");
            if(phase=="cancel")manager.reset();else manager->finish_window(0);
        }
        bool rejected=false;
        if(manager)try {manager->apply_pending();}catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected direct exchange failure")throw;
            rejected=true;
        }
        const bool injected=phase=="direct_backup" || phase=="direct_backup_batch" || phase=="direct_refill" || phase=="upload" ||
            phase=="direct_refill_batch" || phase=="map" || phase=="host" || phase=="publish";
        if(rejected!=injected)throw std::runtime_error("direct exchange failure was not propagated");
        int resident=(phase=="none" || phase=="window" || phase=="prefill")?1:0;
        if(manager && resident==1 && (!manager->stats().direct_exchange_bytes || manager->stats().direct_exchange_rounds!=1))
            throw std::runtime_error("direct exchange fixture did not execute the candidate");
        if(manager && resident==1 && batching && (cached_projections || gpu_wire) && !manager->stats().cached_copy_batches)
            throw std::runtime_error("direct exchange fixture did not batch cached refills");
        if(resident==1 && gpu_wire && cached_projections<3 && !MoeResidencyTestAccess::wire_payload_used(*cache))
            throw std::runtime_error("compact incoming weights did not use available GPU wire");
        if(phase=="reader" && !reader)throw std::runtime_error("direct exchange did not reject the arriving reader");
        reader.reset();
        const auto compare_all=[&] {
            MoeResidencyTestAccess::check_exchange_wire(*cache);
            for(int p=0;p<3;++p)for(int expert=0;expert<3;++expert) {
                if(sources[p]->gpu_resident(expert)!=(expert==resident))
                    throw std::runtime_error("direct exchange ownership differs after completion or rollback");
                auto original=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")))->read_expert(expert);
                if(expert==resident)MoeResidencyTestAccess::compare_gpu(*cache,p,expert,original);
                else {
                    auto held=sources[p]->acquire_expert(expert);
                    const auto actual=moe_cache_fields(held->weights),expected=moe_cache_fields(original);
                    for(std::size_t f=0;f<actual.size();++f) {
                        const auto size=std::size_t(actual[f].numel()*actual[f].element_size());
                        if(size && std::memcmp(actual[f].data_ptr(),expected[f].data_ptr(),size))
                            throw std::runtime_error("direct exchange changed a RAM weight byte");
                    }
                }
                if(sources[p]->expert_disk_reads_after_preload())throw std::runtime_error("direct exchange read sealed storage");
            }
        };
        compare_all();
        if(phase=="none") {
            for(int p=0;p<cached_projections;++p)MoeResidencyTestAccess::seed(*cache,p,2);
            manager->after_layer(0,std::vector<int32_t>(70,2),70);manager->apply_pending();
            resident=2;compare_all();
            if(manager->stats().direct_exchange_rounds!=2)throw std::runtime_error("direct exchange failed to repeat with new fields");
            ++repeats;
        }
        ++cases;
    }
    std::cout<<"direct_exchange_cases="<<cases<<" repeat_cases="<<repeats
        <<" all RAM/GPU bytes, partial backup/refill failures, map/host/publication rollback, reader exclusion, delayed publication and cancellation exact PASS\n";
}
