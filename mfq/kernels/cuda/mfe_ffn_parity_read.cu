#define MFQ_MFE_COMPACT_NVQ 1
#define MFQ_MFE_PARITY_READ 1
#include "mfe_ffn_device.cuh"
#include "mfe_ffn_gu_prequantized.cuh"
#include "mfe_ffn_gu_transfer_routes.cuh"
#include <atomic>
#include <cstdio>

namespace mfq::cuda {
namespace {
template<uint32_t Formats>
void parity_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    partition|=16;
    const auto enabled=[](const char* name){const auto* p=std::getenv(name);return !p || p[0]!='0';};
    if(b.plan_ready && b.resident_plan_overlap && enabled("MFQ_MFE_EARLY_GU"))partition|=8;
    if(b.shared && enabled("MFQ_MFE_SHARED_FIRST"))partition|=4;
    const auto grid=dim3((width+7)/8,b.tokens*(b.routes+int(b.shared!=nullptr)));
    if(!b.completion && enabled("MFQ_MFE_GU_PREQUANTIZED")) {
        const int tiles=b.routes*((b.intermediate+7)/8)+(b.shared?(b.shared_intermediate+7)/8:0);
        const bool simple_m1=b.tokens==1 && (!b.shared || (b.intermediate+7)/8==(b.shared_intermediate+7)/8);
        if(simple_m1)mfe_ffn_gu_prequantized_kernel<Formats,true><<<grid,dim3(32,8),0,stream>>>(b,partition);
        else mfe_ffn_gu_prequantized_kernel<Formats,false><<<dim3(tiles,b.tokens),dim3(32,8),0,stream>>>(b,partition);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_gu_prequantized_dispatch scope=first_gu_launch enabled=1 tokens=%d partition=%d mask=%u packed_grid=%d simple_m1=%d route_reads=1 launched_blocks=%d old_blocks=%d\n",
            b.tokens,partition,b.nvq_format_mask,int(!simple_m1),int(simple_m1),tiles*b.tokens,int(grid.x*grid.y));
    }else mfe_ffn_parallel_gate_up_kernel<2,true,Formats,true,false>
        <<<grid,dim3(32,8),0,stream>>>(b,partition);
}
}
// Called only after the ordinary compact launcher proves its full contract.
void mfe_ffn_parity_read_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0x3f820u))parity_gate_up<0x3f820u>(b,partition,width,stream);
    else parity_gate_up<0x3f922u>(b,partition,width,stream);
}
void mfe_ffn_parity_transfer_routes(const MfeFfnBatch& b,const int32_t* routes,cudaStream_t stream) {
    const auto grid=dim3((b.intermediate+7)/8,std::min(b.routes,4));
    if(!(b.nvq_format_mask&~0x3f820u))mfe_ffn_gu_transfer_routes_kernel<0x3f820u>
        <<<grid,dim3(32,8),0,stream>>>(b,routes);
    else mfe_ffn_gu_transfer_routes_kernel<0x3f922u>
        <<<grid,dim3(32,8),0,stream>>>(b,routes);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_transfer_route_list_dispatch scope=first_transfer_gu enabled=1 tokens=1 workers=%u blocks=%u routes=%d\n",
        grid.y,grid.x*grid.y,b.routes);
}
}
