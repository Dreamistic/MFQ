#pragma once
#include "dma_copy_batch.h"
#include <cuda.h>
#include <array>
#include <cstddef>

namespace mfq::cuda {
// CUDA 13 Driver API, resolved by ABI version while retaining CUDA 12 builds.
// https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__MEM.html
// The Windows source-alias requirement follows Strata's MIT dma_batch.hpp.
class NativeDmaBatch {
    struct Attributes {
        int src_access_order=1; // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        CUmemLocation src_location{},dst_location{};
        unsigned flags=0;
    };
    static_assert(sizeof(Attributes)==24 && offsetof(Attributes,flags)==20);
    using Function=CUresult(CUDAAPI*)(CUdeviceptr*,CUdeviceptr*,std::size_t*,std::size_t,
        Attributes*,std::size_t*,std::size_t,CUstream);
    static Function entry() {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
        static const auto function=[]() -> Function {
            void* raw=nullptr;
            cudaDriverEntryPointQueryResult query=cudaDriverEntryPointSymbolNotFound;
            const auto result=cudaGetDriverEntryPoint("cuGetProcAddress",&raw,cudaEnableDefault,&query);
            if(result==cudaErrorNotSupported){cudaGetLastError();return nullptr;}
            if(result!=cudaSuccess)throw std::runtime_error("CUDA driver entry query failed: "+std::to_string(result));
            if(query!=cudaDriverEntryPointSuccess || !raw)return nullptr;
            const auto get=reinterpret_cast<decltype(&cuGetProcAddress)>(raw);
            void* batch=nullptr;CUdriverProcAddressQueryResult status=CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            if(get("cuMemcpyBatchAsync",&batch,13000,CU_GET_PROC_ADDRESS_DEFAULT,&status)!=CUDA_SUCCESS ||
                    status!=CU_GET_PROC_ADDRESS_SUCCESS)return nullptr;
            return reinterpret_cast<Function>(batch);
        }();
        return function;
#else
        return nullptr;
#endif
    }
public:
    static bool available(){return entry()!=nullptr;}
    // Every source has a live pinned owner; callers translate its registered
    // base to the device-visible alias once, before the consumer graph starts.
    template<class Alias>
    static bool submit(const std::vector<DmaCopy>& copies,cudaStream_t stream,int mode,Alias&& alias) {
        if(!mode || copies.size()<2 || copies.size()>128 || !entry())return false;
        std::array<CUdeviceptr,128> destinations{},sources{};
        std::array<std::size_t,128> sizes{};
        for(std::size_t i=0;i<copies.size();++i) {
            const auto& copy=copies[i];
            if(!copy.destination || !copy.source || !copy.bytes)
                throw std::invalid_argument("empty native DMA field");
            sources[i]=alias(copy.source,copy.bytes);
            if(!sources[i])return false;
            destinations[i]=reinterpret_cast<CUdeviceptr>(copy.destination);sizes[i]=copy.bytes;
        }
        Attributes attributes;attributes.flags=mode>=2?1u:0u;
        std::size_t first=0;
        const auto result=entry()(destinations.data(),sources.data(),sizes.data(),copies.size(),
            &attributes,&first,1,reinterpret_cast<CUstream>(stream));
        if(result==CUDA_ERROR_NOT_SUPPORTED)return false;
        if(result!=CUDA_SUCCESS) {
            std::string detail="CUDA native DMA batch failed: "+std::to_string(result)+" copies="+std::to_string(copies.size());
            for(std::size_t i=0;i<std::min<std::size_t>(copies.size(),12);++i) {
                cudaPointerAttributes src{},dst{};
                const auto source_status=cudaPointerGetAttributes(&src,copies[i].source);
                const auto destination_status=cudaPointerGetAttributes(&dst,copies[i].destination);
                detail+=" ["+std::to_string(i)+" bytes="+std::to_string(sizes[i])+" src="+std::to_string(sources[i])+
                    " dst="+std::to_string(destinations[i])+" type="+std::to_string(int(src.type))+"/"+
                    std::to_string(int(dst.type))+" query="+std::to_string(source_status)+"/"+std::to_string(destination_status)+"]";
            }
            throw std::runtime_error(detail);
        }
        return true;
    }
};
}
