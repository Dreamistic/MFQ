#pragma once
#include <cmath>
#include <limits>

static void sampling_key_variant(int variant) {
    const auto set=[](const char* name,const char* value) {
#ifdef _WIN32
        if(_putenv_s(name,value))throw std::runtime_error("sampling environment setting failed");
#else
        if(setenv(name,value,1))throw std::runtime_error("sampling environment setting failed");
#endif
    };
    set("MFQ_SAMPLE_RADIX_TOPK","1");
    set("MFQ_SAMPLE_RADIX_KEYS",variant?"1":"0");
    set("MFQ_SAMPLE_RADIX_ITEMS",variant==17?"1":variant==18?"4":variant>=8?"8":variant==4?"4":"1");
    set("MFQ_SAMPLE_KEY_BITONIC",variant>=16?"1":"0");
}

static void benchmark_sampling_keys(bool timings=true) {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    int checks=0;
    for(int vocab:{7,31,255,256,257,1009,151936,248320})
    for(int top_k:{1,2,20,64})if(top_k<=vocab)
    for(auto dtype:{tb::kFloat32,tb::kFloat16,tb::kBFloat16}) {
        auto host=tb::empty({1,vocab},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto logits=tb::zeros({1,vocab},tb::TensorOptions().device(tb::kCUDA).dtype(dtype));
        auto random=tb::zeros({1},logits.options().dtype(tb::kFloat32));
        for(double p:{0.0,0.8,1.0})for(int variant:{1,4,8,16,17,18}) {
            sampling_key_variant(variant);
            tb::Tensor actual;
            const auto body=[&]{actual={};actual=sample_top_k_top_p_cuda(logits,random,0.7,top_k,p);};
            {GraphWarmupScope warmup(context,stream);body();}
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));actual={};
            Graph graph;graph.capture_begin_shared();body();graph.capture_end();
            for(int step=0;step<4;++step) {
                for(int i=0;i<vocab;++i)host.data_ptr<float>()[i]=step==0?0.f:
                    step==1?float((i*37)%113-56)*0.125f:std::sin(float(i)*.037f)*5.f;
                if(step==2) {
                    host.data_ptr<float>()[0]=std::numeric_limits<float>::quiet_NaN();
                    host.data_ptr<float>()[1]=-std::numeric_limits<float>::infinity();
                    host.data_ptr<float>()[2]=-0.f;
                }
                if(step==3)host.data_ptr<float>()[vocab-1]=std::numeric_limits<float>::infinity();
                logits.copy_(host.to(tb::kCUDA).to(dtype));
                random.fill_(step==0?0.f:step==1?.137f:step==2?.731f:1.f);
                sampling_key_variant(0);
                auto expected=sample_top_k_top_p_cuda(logits,random,0.7,top_k,p).item<int64_t>();
                graph.replay();
                check(actual.item<int64_t>()==expected,"packed-key sampling changed the sampled token");
                ++checks;
            }
        }
    }
    if(timings)for(int vocab:{256,4096,32768,151936,248320})
    for(int top_k:{20,64})if(top_k==20 || vocab>=151936)
    for(int variant:{0,16,17,18,0}) {
        auto host=tb::empty({1,vocab},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int i=0;i<vocab;++i)host.data_ptr<float>()[i]=std::sin(float(i)*.037f)*5.f;
        auto logits=host.to(tb::kCUDA),random=tb::full({1},0.371,logits.options());
        sampling_key_variant(variant);
        tb::Tensor output;
        const auto body=[&]{output={};output=sample_top_k_top_p_cuda(logits,random,0.7,top_k,0.8);};
        {GraphWarmupScope warmup(context,stream);body();}
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));output={};
        Graph graph;graph.capture_begin_shared();for(int i=0;i<16;++i)body();graph.capture_end();
        for(int i=0;i<8;++i)graph.replay();
        cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<64;++i)graph.replay();
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
        std::cout<<"sampling_keys vocab="<<vocab<<" top_k="<<top_k<<" variant="<<variant<<" us="<<ms*1000/(64*16)<<std::endl;
    }
    std::cout<<"Sampling packed-key graph cases="<<checks<<" PASS"<<std::endl;
}
