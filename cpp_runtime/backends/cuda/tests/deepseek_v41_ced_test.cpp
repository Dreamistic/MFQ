#include "models/deepseek_v41/ops.h"
#include <iostream>

namespace {
namespace tb=mfq_tensor_backend;
using namespace mfq::cuda;
using namespace mfq::cuda::deepseek_v41_runtime;

void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}
tb::Tensor fp32(std::vector<int64_t> shape) {
    return tb::ones(shape,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
}
QuantLinear dense(int out,int in) {
    QuantLinear linear;
    linear.kind=QuantLinearKind::Dense;
    linear.dense=(tb::randn({out,in},fp32({1}).options())/std::sqrt(double(in)))
        .to(tb::kFloat16).contiguous();
    return linear;
}

DeepseekV41CausalLm make_model(CudaExecutionContext& execution) {
    tb::manual_seed(91);
    DeepseekV41CausalLm model;
    model.execution=&execution;
    auto& c=model.config;
    c.hidden=64;c.vocab=8;c.n_layers=4;c.causal_encoder_layers=2;
    c.hc_mult=4;c.hc_sinkhorn_iters=20;c.hc_eps=1.e-6;
    c.n_heads=64;c.n_kv_heads=1;c.head_dim=512;c.rope_head_dim=64;
    c.q_lora_rank=32;c.o_groups=8;c.o_lora_rank=8;
    c.index_n_heads=32;c.index_head_dim=128;c.index_topk=16;
    c.candidate_source_layer=2;c.candidate_block_size=8;c.candidate_topk_blocks=8;
    c.compress_ratios={0,2,1,1};c.kv_source_layers={1,2};c.index_source_layers={1,2,3};
    c.sliding_window=128;c.max_position_embeddings=1024;c.dspark_target_layer_ids={2,3};
    model.metadata.hidden_size=c.hidden;model.metadata.vocab_size=c.vocab;
    model.metadata.num_hidden_layers=c.n_layers;model.metadata.hc_mult=4;
    model.metadata.max_position_embeddings=c.max_position_embeddings;
    model.metadata.rms_norm_eps=c.rms_eps;
    model.embed=dense(c.vocab,c.hidden);model.output_norm=fp32({c.hidden});
    model.shared=std::make_shared<SharedState>();model.shared->config=c;
    model.shared->engram_hash=std::make_unique<EngramHashState>(8,3,2,0,
        std::vector<int64_t>{1},std::vector<int64_t>{36},std::vector<int64_t>{5,7,11,13},
        std::vector<int64_t>{0,5,12,23},std::vector<int64_t>{3,5,7},
        std::vector<int>{0,1,2,3,4,5,6,7});
    for(int layer=0;layer<c.n_layers;++layer) {
        auto block=std::make_unique<deepseek_v41_runtime::Block>();
        auto& b=*block;
        b.config=c;b.shared=model.shared;b.layer=layer;b.ratio=c.compress_ratios[layer];
        b.max_context=c.max_position_embeddings;b.attention_norm=b.mlp_norm=fp32({c.hidden});
        b.query_a_norm=fp32({c.q_lora_rank});b.key_value_norm=b.compressor_norm=fp32({512});
        b.index_key_norm=fp32({128});b.sinks=tb::zeros({64},fp32({1}).options());
        b.attention_mhc_function=tb::randn({24,4*c.hidden},fp32({1}).options())*.03;
        b.mlp_mhc_function=tb::randn({24,4*c.hidden},fp32({1}).options())*.03;
        b.attention_mhc_scale=b.mlp_mhc_scale=fp32({3});
        b.attention_mhc_base=b.mlp_mhc_base=tb::zeros({24},fp32({1}).options());
        b.query_a=dense(c.q_lora_rank,c.hidden);b.query_b=dense(64*512,c.q_lora_rank);
        b.key_value=dense(512,c.hidden);b.output_a=dense(8*8,64*512/8);b.output_b=dense(c.hidden,8*8);
        b.compressor_key_value=dense(512,c.hidden);b.compressor_gate=dense(512,c.hidden);
        b.index_key=dense(128,512);b.index_query=dense(32*128,c.q_lora_rank);b.index_score=dense(32,c.hidden);
        b.mlp.gate_up.layers.push_back(dense(128,c.hidden));
        b.mlp.gate_up.layers.push_back(dense(128,c.hidden));b.mlp.down=dense(c.hidden,128);
        b.rope=Dsv4RopeTable(c.max_position_embeddings,c.rope_theta,b.ratio,c.compress_rope_theta);
        model.blocks.push_back(std::move(block));
    }
    model.reset(1);
    return model;
}
tb::Tensor ids(int tokens) {
    return tb::arange(tokens,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64))
        .remainder(8.).to(tb::kInt64).reshape({1,tokens}).contiguous();
}
void check_positions(const DeepseekV41CausalLm& model,int64_t expected) {
    require(model.cache_pos==expected,"CED logical cache position changed");
    for(const auto& block:model.blocks) {
        const auto& b=dynamic_cast<const deepseek_v41_runtime::Block&>(*block);
        require(b.state.position==expected,"CED physical cache position changed");
        if(b.kv_source())require(b.state.compressed_length==expected/b.ratio,
                                "CED duplicated or lost global KV rows");
    }
}
void check_exact_short() {
    CudaExecutionContext execution;
    auto model=make_model(execution);
    auto input=ids(17);
    auto expected=model.hidden_forward(input);
    model.reset(1);
    auto actual=deepseek_v41_prefill_hidden(model,input,17);
    require(actual.sizes()==expected.sizes(),"CED short prompt geometry changed");
    const auto error=(actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<double>();
    std::cerr<<"CED short maximum error="<<error<<'\n';
    require(std::isfinite(error) && error<.003,"CED short prompt differs from complete forward");
    check_positions(model,17);
    model.hidden_forward(ids(1));check_positions(model,18);
}
void check_long_and_cancel() {
    CudaExecutionContext execution;
    auto model=make_model(execution);
    tb::Tensor raw;
    auto result=deepseek_v41_prefill_hidden(model,ids(257),64,{},&raw);
    require(raw.sizes()==tb::IntArrayRef({1,128,2*64}),"CED DSpark target tail geometry changed");
    require(result.size(1)==128,"CED did not limit Decoder replay to the local window");
    require(std::isfinite(result.to(tb::kFloat32).abs().max().item<double>()),"CED output is not finite");
    check_positions(model,257);
    model.hidden_forward(ids(1));check_positions(model,258);
    deepseek_v41_prefill_hidden(model,ids(3),2);check_positions(model,261);
    int calls=0;
    auto cancelled=deepseek_v41_prefill_hidden(model,ids(129),32,[&]{return ++calls>=4;});
    require(!cancelled.defined(),"CED cancellation returned partial output");
    check_positions(model,0);
}
}
int main() try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    check_exact_short();check_long_and_cancel();run_dspark_self_check();
    std::cout<<"DeepSeek-V4.1 CED numerical/cache tests passed\n";return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
