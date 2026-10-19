#include "mx.h"
#include "mfq_cuda_quant_ops.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace tb=mfq_tensor_backend;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
float decode(uint8_t raw) {
    const int exponent=(raw>>3)&15,mantissa=raw&7;
    const float magnitude=exponent?std::ldexp(1+float(mantissa)/8,exponent-7):std::ldexp(float(mantissa),-9);
    return raw&128?-magnitude:magnitude;
}
void verify(const tb::Tensor& value,const std::vector<float>& expected) {
    const auto actual=value.to(tb::kCPU).to(tb::kFloat32).contiguous();
    require(actual.numel()==expected.size(),"MXFP8 output shape differs");
    for(std::size_t i=0;i<expected.size();++i)
        if(!std::isfinite(actual.data_ptr<float>()[i]) ||
            std::abs(actual.data_ptr<float>()[i]-expected[i])>2e-4f+1e-3f*std::abs(expected[i]))
            throw std::runtime_error("MXFP8 independent oracle mismatch at "+std::to_string(i));
}
void check(int block_rows,int block_columns,int outputs,int width,int groups=1) {
    std::vector<uint8_t> blob{'M','X','T','1',1,8,0,0};
    const int scale_rows=(outputs+block_rows-1)/block_rows,scale_columns=width/block_columns;
    for(uint64_t d:{uint64_t(outputs),uint64_t(width),uint64_t(outputs),uint64_t(width),
            uint64_t(scale_rows),uint64_t(scale_columns)}) {
        const auto at=blob.size();blob.resize(at+8);std::memcpy(blob.data()+at,&d,8);
    }
    for(int i=0;i<outputs*width;++i) {
        const auto raw=uint8_t((i*37+11)%240);blob.push_back((raw&127)==127?uint8_t(raw-1):raw);
    }
    for(int i=0;i<scale_rows*scale_columns;++i)blob.push_back(uint8_t(119+(i*7)%5));
    auto cpu=unpack_mxfp8(blob);
    require(cpu.block_rows==block_rows && cpu.block_columns==block_columns,"MXFP8 scale block was discarded");
    auto weight=to_device_mxfp8(cpu,true);
    std::vector<float> dense(outputs*width);
    for(int n=0;n<outputs;++n)for(int k=0;k<width;++k)
        dense[std::size_t(n)*width+k]=float(mfq_half(decode(cpu.values[std::size_t(n)*width+k])*
            std::ldexp(1.0f,int(cpu.scales[(n/block_rows)*scale_columns+k/block_columns])-127)));
    verify(mxfp8_dequant_cuda(weight.values,weight.scales),dense);
    verify(mxfp8_cpu_reference(weight),dense);
    CudaProfiler profiler;
    for(int m:{1,6,17,129}) {
        std::vector<float> input(std::size_t(m)*groups*width),expected(std::size_t(m)*outputs);
        for(std::size_t i=0;i<input.size();++i)input[i]=float(int(i*13%17)-8)/32;
        const int per=outputs/groups;
        for(int row=0;row<m;++row)for(int n=0;n<outputs;++n) {
            double sum=0;
            for(int k=0;k<width;++k)sum+=double(input[(std::size_t(row)*groups+n/per)*width+k])*dense[std::size_t(n)*width+k];
            expected[std::size_t(row)*outputs+n]=float(sum);
        }
        auto x=tb::tensor(input).reshape({m,groups,width}).to(tb::kFloat16).to(tb::kCUDA);
        if(groups==1) {
            x=x.reshape({m,width});
            verify(mxfp8_matmul(profiler,weight,x),expected);
            verify(mxfp8_matmul_f32(profiler,weight,x),expected);
            verify(mxfp8_matmul(profiler,to_device_mxfp8(cpu,false),x.to(tb::kCPU)),expected);
        } else {
            verify(mxfp8_groupwise_matmul(profiler,weight,x,groups),expected);
            verify(mxfp8_groupwise_matmul_f32(profiler,weight,x,groups),expected);
        }
    }
    auto input_slice=slice_mxfp8_cpu(cpu,TensorParallelAxis::Input,block_columns,width);
    require(input_slice.block_columns==block_columns && input_slice.block_rows==block_rows &&
        input_slice.scales.size()==std::size_t(scale_rows)*(scale_columns-1),"MXFP8 input slice scale geometry changed");
    const int first=block_rows==1?3:block_rows;
    auto output_slice=slice_mxfp8_cpu(cpu,TensorParallelAxis::Output,first,outputs);
    require(output_slice.scales.size()==std::size_t((outputs-first+block_rows-1)/block_rows)*scale_columns,
        "MXFP8 output slice scale geometry changed");
    std::cout<<"MXFP8 "<<block_rows<<'x'<<block_columns<<" outputs="<<outputs<<" width="<<width<<" groups="<<groups<<" PASS\n";
}
}
int main() try {
    check(128,128,137,256);
    check(32,32,65,96);
    check(1,32,65,96);
    check(32,32,96,96,3);
    check(1,32,96,96,3);
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
