#pragma once
#include "mfe_ffn.h"

namespace mfq::cuda {
__device__ __forceinline__ float mfe_mxfp4_scale(uint8_t code) {
    if(code==255)return __int_as_float(0x7fffffff);
    return code==0?__int_as_float(0x00400000):__int_as_float(unsigned(code)<<23);
}

// MXFP4 consumes the original FP16 activation. NINT/NVQ input quantization
// buffers must never substitute for that activation in a mixed expert bundle.
__device__ __forceinline__ void mfe_mxfp4_pair(const MfePackedProjection& weight,
        int local,int first,const __half* input,__half* result) {
    if(!input) {
        result[0]=result[1]=__float2half(__int_as_float(0x7fffffff));return;
    }
    constexpr float magnitude[8]={0,.5f,1,1.5f,2,3,4,6};
    const int lane=int(threadIdx.x)&31,width=weight.input_width;
    const auto* packed=static_cast<const uint8_t*>(weight.fields[0]);
    const auto* scales=static_cast<const uint8_t*>(weight.fields[1]);
    float sums[2]{};
    for(int column=lane;column<width;column+=32) {
        const float x=__half2float(input[column]);
#pragma unroll
        for(int row=0;row<2;++row)if(first+row<weight.output_rows) {
            const int64_t neuron=int64_t(local)*weight.output_rows+first+row;
            const auto byte=packed[neuron*(width/2)+column/2];
            const unsigned code=(byte>>((column&1)*4))&15;
            const float value=(code&8?-magnitude[code&7]:magnitude[code&7])*
                mfe_mxfp4_scale(scales[neuron*(width/32)+column/32]);
            sums[row]+=x*value;
        }
    }
#pragma unroll
    for(int offset=16;offset;offset>>=1) {
        sums[0]+=__shfl_down_sync(0xffffffffu,sums[0],offset);
        sums[1]+=__shfl_down_sync(0xffffffffu,sums[1],offset);
    }
    result[0]=__float2half_rn(__shfl_sync(0xffffffffu,sums[0],0));
    result[1]=__float2half_rn(__shfl_sync(0xffffffffu,sums[1],0));
}
}
