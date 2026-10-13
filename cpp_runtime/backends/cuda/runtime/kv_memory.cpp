#include "kv_memory.h"
#include "kv_vmm.h"
#include "execution_options.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
constexpr const char* insufficient = "kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length";
std::size_t setting(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const auto result = std::strtoull(value, &end, 10);
    if (*value == '-' || !end || *end) throw std::invalid_argument(std::string("invalid ") + name);
    return std::size_t(result);
}
std::size_t host_available() {
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? std::size_t(status.ullAvailPhys) : 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    return pages > 0 && size > 0 ? std::size_t(pages) * size : 0;
#endif
}
std::size_t width_bytes(const tb::TensorOptions& options) { return options.scalar_type() == tb::kInt32 ? 4 : 2; }
tb::Tensor external(void* data, std::shared_ptr<void> owner, const std::vector<int64_t>& shape,
                    const tb::TensorOptions& options) {
    auto view = make_contiguous_view(data, shape, options.scalar_type(), options.target_device());
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data; storage->bytes = view.nbytes(); storage->device = options.target_device();
    storage->owner = std::move(owner);
    return Tensor(std::move(storage), view);
}
tb::Tensor host_tensor(const std::vector<int64_t>& shape, const tb::TensorOptions& options) {
    std::size_t bytes = width_bytes(options);
    for (const auto dimension : shape) {
        if (dimension <= 0 || std::size_t(dimension) > std::numeric_limits<std::size_t>::max()/bytes)
            throw std::invalid_argument("KV tensor size overflow");
        bytes *= dimension;
    }
    void* host = nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    charge_tensor_host_bytes(bytes);
    auto owner = std::shared_ptr<void>(host, [bytes](void* p) {
        (void)cudaDeviceSynchronize(); (void)cudaFreeHost(p); tensor_host_bytes.fetch_sub(bytes);
    });
    std::memset(host, 0, bytes);
    void* gpu = nullptr; MFQ_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&gpu, host, 0));
    return external(gpu, std::move(owner), shape, options);
}
}

struct KvMemoryAllocation::Virtual {
    kv_vmm::VmmRange range;
    int64_t maximum, heads, width, committed = 0;
    std::size_t element;
    Virtual(int64_t maximum, int64_t heads, int64_t width, std::size_t element)
        : maximum(maximum), heads(heads), width(width), element(element) {
        if (!range.reserve(std::size_t(maximum)*heads*width*element))
            throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory reservation failed");
    }
    ~Virtual() { (void)cudaDeviceSynchronize(); }
    std::set<int64_t> missing(int64_t tokens) const {
        std::set<int64_t> result;
        tokens=std::min(maximum,((tokens+8191)/8192)*8192);
        if (tokens <= committed) return result;
        const auto g = kv_vmm::vmm_granularity();
        for (int64_t head=0; head<heads; ++head) {
            const auto begin = std::size_t(head)*maximum*width*element;
            const auto end = begin + std::size_t(tokens)*width*element;
            for (auto page=begin/g; page<(end+g-1)/g; ++page)
                if (!range.mapped(int64_t(page))) result.insert(int64_t(page));
        }
        return result;
    }
    void ensure(int64_t tokens) {
        tokens=std::min(maximum,((tokens+8191)/8192)*8192);
        const auto fresh = missing(tokens);
        if (!fresh.empty()) MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        for (const auto page : fresh) {
            if (!range.map_range(page,page+1,[]{return kv_vmm::VmmChunk(0);}))
                throw std::runtime_error(insufficient);
            MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(range.base()+page*kv_vmm::vmm_granularity(),0,
                kv_vmm::vmm_granularity(),current_stream().stream()));
        }
        committed = std::max(committed,tokens);
    }
};

KvMemoryOptions kv_memory_options() {
    KvMemoryOptions o;
    if (const auto* p=std::getenv("MFQ_KV_PLACEMENT")) o.placement=p;
    if (o.placement!="auto" && o.placement!="auto_ssd" && o.placement!="vram" && o.placement!="ram")
        throw std::invalid_argument("invalid KV placement");
    if (const auto* p=std::getenv("MFQ_KV_ALLOCATION")) {
        const std::string allocation=p;
        if (allocation!="preallocated" && allocation!="dynamic") throw std::invalid_argument("invalid KV allocation");
        o.dynamic=allocation=="dynamic";
    }
    if (o.dynamic && (o.placement=="ram" || o.placement=="auto_ssd"))
        throw std::invalid_argument("kv_dynamic_unavailable: experimental dynamic KV requires VRAM or automatic without SSD");
    o.gpu_limit=setting("MFQ_KV_GPU_LIMIT_BYTES",0);
    o.ram_limit_set=std::getenv("MFQ_KV_RAM_LIMIT_BYTES")!=nullptr;
    o.ram_limit=setting("MFQ_KV_RAM_LIMIT_BYTES",0);
    if (const auto* p=std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) o.directory=p;
    return o;
}
std::size_t KvMemoryAllocation::raw_bytes() const {return std::size_t(maximum)*heads*(key_width+value_width)*width_bytes(options);}
std::size_t KvMemoryAllocation::index_bytes() const {return std::size_t(maximum+(maximum+pool-1)/pool)*index_width*2;}
std::size_t KvMemoryAllocation::gpu_bytes() const {
    if (!ready) return 0;
    if (virtual_keys) return index_bytes() + std::size_t(virtual_keys->range.mapped_count()+virtual_values->range.mapped_count())*kv_vmm::vmm_granularity();
    return index_bytes() + (host || offload ? 0 : raw_bytes());
}
std::size_t KvMemoryAllocation::ram_bytes() const {return ready && host ? raw_bytes() : 0;}
std::size_t KvMemoryAllocation::growth_bytes(int64_t tokens) const {
    tokens=std::min(maximum,tokens);
    return virtual_keys ? (virtual_keys->missing(tokens).size()+virtual_values->missing(tokens).size())*kv_vmm::vmm_granularity() : 0;
}
void KvMemoryAllocation::ensure(int64_t tokens) {
    if (tokens<0 || tokens>maximum) throw std::invalid_argument("KV growth exceeds configured context");
    if (virtual_keys) {virtual_keys->ensure(tokens); virtual_values->ensure(tokens);}
}
KvMemoryManager::KvMemoryManager(KvMemoryOptions options):options_(std::move(options)) {}
std::shared_ptr<KvMemoryAllocation> KvMemoryManager::add(int64_t maximum,int64_t heads,int64_t kw,int64_t vw,
        int64_t iw,int64_t pool,const tb::TensorOptions& options) {
    if (ready_) {
        auto policy=options_;policy.additional_request=true;
        // Independent graph sessions require directly addressable raw caches.
        // Admit extra GPU sessions against currently free memory.
        if(policy.placement=="auto" || policy.placement=="auto_ssd")policy.placement="vram";
        KvMemoryManager child(policy);
        auto a=child.add(maximum,heads,kw,vw,iw,pool,options);
        child.finalize();allocations_.push_back(a);return a;
    }
    auto a=std::make_shared<KvMemoryAllocation>();
    a->maximum=maximum;a->heads=heads;a->key_width=kw;a->value_width=vw;a->index_width=iw;a->pool=pool;a->options=options;
    allocations_.push_back(a);return a;
}
void KvMemoryManager::finalize(std::size_t minimum_experts) {
    if (ready_) return;
    std::size_t raw=0,index=0;
    for (auto& weak:allocations_) if(auto a=weak.lock()){raw+=a->raw_bytes();index+=a->index_bytes();}
    if (!raw) {ready_=true;return;}
    auto context=default_context();context->reclaim_idle_memory();
    const auto free=context->available_memory();
    const auto reserve=(options_.additional_request?std::size_t(256)<<20:runtime_options::workspace_reserve_bytes())+minimum_experts;
    if (free<reserve || index>free-reserve) throw std::runtime_error(insufficient);
    auto gpu=free-reserve-index;
    if(options_.gpu_limit)gpu=std::min(gpu,options_.gpu_limit);
    auto available=host_available();
    auto ram=available> (1ULL<<30) ? available-(1ULL<<30) : 0;
    if(options_.ram_limit_set)ram=std::min(ram,options_.ram_limit);
    if(options_.placement=="vram" && raw>gpu)throw std::runtime_error("kv_vram_insufficient: configured KV cache does not fit in available VRAM");
    if(options_.placement=="ram" && raw>ram)throw std::runtime_error(insufficient);
    if(options_.placement=="auto" && (raw>gpu && raw-gpu>ram))throw std::runtime_error(insufficient);
    if(options_.dynamic && !kv_vmm::vmm_available())throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory management is unavailable");
    if(options_.dynamic && raw>gpu)throw std::runtime_error("kv_dynamic_unavailable: experimental dynamic KV currently requires a GPU-resident cache");
    if(!options_.dynamic && options_.placement!="ram" && raw>gpu) {
        KvOffloadConfig config;
        config.buffer_bytes=std::min<std::size_t>(64ULL<<20,std::max<std::size_t>(4096,raw/16));
        // Streaming arenas are preallocated. SSD is opened only on first spill.
        config.gpu_budget_bytes=std::min(raw,gpu);
        if(config.gpu_budget_bytes<config.buffer_bytes*2)throw std::runtime_error(insufficient);
        config.ram_budget_bytes=std::min(ram,raw-(config.gpu_budget_bytes-config.buffer_bytes));
        // A partially filled block is replaced transactionally during append.
        // Retain room for that replacement in addition to the complete cache.
        const auto required_ram=raw-(config.gpu_budget_bytes-config.buffer_bytes)+config.buffer_bytes;
        if(options_.placement=="auto" && required_ram>ram)throw std::runtime_error(insufficient);
        config.ram_budget_bytes=std::min(ram,required_ram);
        config.directory=options_.directory;config.preallocate=true;config.ssd_enabled=options_.placement=="auto_ssd";
        offload_=std::make_shared<KvOffloadStore>(config);
    }
    ResidentAllocationScope resident;
    for(auto& weak:allocations_)if(auto a=weak.lock()) {
        const auto shape=[&](int64_t w){return std::vector<int64_t>{1,a->heads,a->maximum,w};};
        auto index_options=a->options;index_options.dtype(tb::kFloat16);
        a->index=tb::zeros({1,a->maximum,a->index_width},index_options);
        a->pooled=tb::zeros({1,(a->maximum+a->pool-1)/a->pool,a->index_width},index_options);
        a->host=options_.placement=="ram";a->offload=offload_;
        if(options_.dynamic) {
            a->virtual_keys=std::make_shared<KvMemoryAllocation::Virtual>(a->maximum,a->heads,a->key_width,width_bytes(a->options));
            a->virtual_values=std::make_shared<KvMemoryAllocation::Virtual>(a->maximum,a->heads,a->value_width,width_bytes(a->options));
            a->keys=external(a->virtual_keys->range.base(),a->virtual_keys,shape(a->key_width),a->options);
            a->values=external(a->virtual_values->range.base(),a->virtual_values,shape(a->value_width),a->options);
            a->ensure(std::min<int64_t>(a->maximum,16384));
        } else if(a->host) {
            a->keys=host_tensor(shape(a->key_width),a->options);a->values=host_tensor(shape(a->value_width),a->options);
        } else if(!offload_) {
            a->keys=tb::zeros(shape(a->key_width),a->options);a->values=tb::zeros(shape(a->value_width),a->options);
        }
        a->ready=true;
    }
    ready_=true;
    std::cerr<<"runtime_kv_memory placement="<<options_.placement<<" allocation="<<(options_.dynamic?"dynamic":"preallocated")
        <<" raw_bytes="<<raw<<" index_bytes="<<index<<" streaming="<<bool(offload_)<<'\n';
}
void KvMemoryManager::ensure(int64_t tokens) {
    std::size_t growth=0;
    for(auto& weak:allocations_)if(auto a=weak.lock())growth+=a->growth_bytes(std::min(a->maximum,tokens));
    if(growth && reserve_)reserve_(growth+(64ULL<<20));
    for(auto& weak:allocations_)if(auto a=weak.lock())a->ensure(std::min(a->maximum,tokens));
}
std::size_t KvMemoryManager::growth_reserve() const {
    std::size_t result=0;
    for(auto& weak:allocations_)if(auto a=weak.lock())result+=a->growth_bytes(a->maximum);
    return result;
}
std::size_t KvMemoryManager::expert_budget_charge() const {
    std::size_t bytes=growth_reserve();
    for(auto& weak:allocations_)if(auto a=weak.lock())bytes+=a->gpu_bytes();
    if(offload_)bytes+=offload_->config().gpu_budget_bytes;
    return bytes;
}
std::vector<std::pair<std::string,double>> KvMemoryManager::metrics() const {
    std::size_t gpu=0,ram=0,capacity=0,vmm=0;
    for(auto& weak:allocations_)if(auto a=weak.lock()){
        gpu+=a->gpu_bytes();ram+=a->ram_bytes();capacity+=a->raw_bytes()+a->index_bytes();
        if(a->virtual_keys)vmm+=a->gpu_bytes()-a->index_bytes();
    }
    return {{"kv_memory_supported",capacity?1.0:0.0},{"kv_memory_dynamic",options_.dynamic?1.0:0.0},
        {"kv_memory_preallocated",options_.dynamic?0.0:1.0},{"kv_memory_capacity_bytes",double(capacity)},
        {"kv_memory_gpu_bytes",double(gpu)},{"kv_memory_ram_bytes",double(ram)},
        {"kv_memory_vmm_bytes",double(vmm)},
        {"kv_memory_ssd_enabled",options_.placement=="auto_ssd"?1.0:0.0}};
}
}
