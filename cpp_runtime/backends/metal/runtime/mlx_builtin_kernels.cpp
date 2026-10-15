#include "mlx_builtin_kernels.h"
#include "mlx_nint_metadata.h"
#include "mfq_mfe_prefill_embedded.h"

namespace mfq::metal {
namespace {
#include "../kernels/mfq_qsa_prefill.inc"
}

std::string qsa_prefill_indexer_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n";
    source += detail::kSteelMmaSource;
    source += "\n#define STEEL_PRAGMA_UNROLL _Pragma(\"clang loop unroll(full)\")\n";
    source += kQsaPrefillIndexerSource;
    return source;
}

std::string sparse_attention_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\nusing bfloat16_t = bfloat;\n";
    source += detail::kSteelAttentionSource;
    source += detail::kDsaSparsePrefillSource;
    source += detail::kSparseBlockGqaSource;
    return source;
}

std::string dsv4_mxfp4_blocks_source() {
    std::string source = "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
        "#define MFQ_ENABLE_DSV4_MXFP4_BLOCKS 1\nusing namespace metal;\n";
    source += detail::kNintMetadataSource;
    source += detail::kSteelMmaSource;
    source += detail::kMfePrefillSource;
    return source;
}

}
