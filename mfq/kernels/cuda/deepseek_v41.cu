#include "mfq/kernels/cuda/deepseek_v41.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <climits>
#include <limits>

namespace {

__global__ void hc_prepare_kernel(const half* residual, const float* previous_pre,
                                  const float* norm, float* normalized, half* branch,
                                  int hidden, float epsilon) {
    const auto row = static_cast<int64_t>(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float* pre = previous_pre + row * 4;
    __shared__ float sums[2][8];
    __shared__ float inverse[2];
    float flat_square = 0, branch_square = 0;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float mixed = 0;
#pragma unroll
        for (int source = 0; source < 4; ++source) {
            const float value = __half2float(residual[(row * 4 + source) * hidden + col]);
            flat_square = __fadd_rn(flat_square, __fmul_rn(value, value));
            mixed = __fadd_rn(mixed, __fmul_rn(value, pre[source]));
        }
        branch_square = __fadd_rn(branch_square, __fmul_rn(mixed, mixed));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        flat_square += __shfl_down_sync(0xffffffff, flat_square, offset);
        branch_square += __shfl_down_sync(0xffffffff, branch_square, offset);
    }
    if (!lane) { sums[0][warp] = flat_square; sums[1][warp] = branch_square; }
    __syncthreads();
    if (!warp) {
        flat_square = lane < 8 ? sums[0][lane] : 0;
        branch_square = lane < 8 ? sums[1][lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            flat_square += __shfl_down_sync(0xffffffff, flat_square, offset);
            branch_square += __shfl_down_sync(0xffffffff, branch_square, offset);
        }
        if (!lane) {
            inverse[0] = rsqrtf(flat_square / (4.0f * hidden) + epsilon);
            inverse[1] = rsqrtf(branch_square / hidden + epsilon);
        }
    }
    __syncthreads();
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float mixed = 0;
#pragma unroll
        for (int source = 0; source < 4; ++source) {
            const auto index = (row * 4 + source) * hidden + col;
            const float value = __half2float(residual[index]);
            normalized[index] = __fmul_rn(value, inverse[0]);
            mixed = __fadd_rn(mixed, __fmul_rn(value, pre[source]));
        }
        branch[row * hidden + col] = __float2half_rn(
            __fmul_rn(__fmul_rn(mixed, inverse[1]), norm[col]));
    }
}

__device__ __forceinline__ float e4m3(float value) {
    const float bounded = fminf(448.0f, fmaxf(-448.0f, value));
    return static_cast<float>(__nv_fp8_e4m3(bounded));
}

__device__ __forceinline__ float pow2_ceil(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const bool has_mantissa = (bits & 0x7fffffu) != 0u;
    return __uint_as_float(
        (exponent + static_cast<std::uint32_t>(has_mantissa)) << 23u);
}

__device__ __forceinline__ float e2m1(float value, float scale) {
    const float normalized = fminf(6.0f, fmaxf(-6.0f, value / scale));
    const float magnitude = fabsf(normalized);
    float quantized = 0.0f;
    if (magnitude <= 0.25f) quantized = 0.0f;
    else if (magnitude < 0.75f) quantized = 0.5f;
    else if (magnitude <= 1.25f) quantized = 1.0f;
    else if (magnitude < 1.75f) quantized = 1.5f;
    else if (magnitude <= 2.5f) quantized = 2.0f;
    else if (magnitude < 3.5f) quantized = 3.0f;
    else if (magnitude <= 5.0f) quantized = 4.0f;
    else quantized = 6.0f;
    return copysignf(quantized * scale, normalized);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void mxfp8_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 32 + lane;
    const float value = __half2float(input[offset]);
    const float maximum = warp_max(fabsf(value));
    const float scale = pow2_ceil(
        fmaxf(maximum, 448.0f * 0x1p-126f) / 448.0f);
    output[offset] = __float2half_rn(e4m3(value / scale) * scale);
}

__global__ void mxfp4_e4m3_scale_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 16 +
        static_cast<std::int64_t>(lane < 16 ? lane : 0);
    const float value = lane < 16 ? __half2float(input[offset]) : 0.0f;
    const float maximum = warp_max(fabsf(value));
    const float raw_scale = fmaxf(maximum, 6.0f * 0x1p-9f) / 6.0f;
    const float scale = fmaxf(e4m3(raw_scale), 0x1p-9f);
    if (lane < 16) {
        output[offset] = __float2half_rn(e2m1(value, scale));
    }
}

void require_input(
    const mfq_tensor_backend::Tensor& input,
    std::int64_t group,
    const char* operation) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == mfq_tensor_backend::kFloat16 &&
            input.dim() >= 1 && input.numel() > 0 &&
            input.size(-1) % group == 0,
        operation,
        ": expected contiguous CUDA f16 with a grouped last dimension");
    MFQ_RUNTIME_CHECK(
        input.numel() / group <= std::numeric_limits<int>::max(),
        operation, ": input is too large");
}

} // namespace

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_prepare_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm, double epsilon) {
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                          residual.scalar_type() == mfq_tensor_backend::kFloat16 &&
                          residual.dim() == 4 && residual.size(2) == 4 &&
                          residual.size(3) > 0 && residual.size(3) <= INT_MAX,
                      "deepseek_v41_hc_prepare: expected CUDA f16 [B,T,4,H]");
    MFQ_RUNTIME_CHECK(previous_pre.is_cuda() && previous_pre.is_contiguous() &&
                          previous_pre.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          previous_pre.dim() == 3 && previous_pre.size(0) == residual.size(0) &&
                          previous_pre.size(1) == residual.size(1) && previous_pre.size(2) == 4 &&
                          norm.is_cuda() && norm.is_contiguous() &&
                          norm.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          norm.numel() == residual.size(3) && std::isfinite(epsilon) && epsilon > 0,
                      "deepseek_v41_hc_prepare: invalid previous weights, norm, or epsilon");
    const auto rows = residual.size(0) * residual.size(1), hidden = residual.size(3);
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= INT_MAX,
                      "deepseek_v41_hc_prepare: row count is out of range");
    auto normalized = mfq_tensor_backend::empty(
        {residual.size(0), residual.size(1), 4 * hidden}, previous_pre.options());
    auto branch = mfq_tensor_backend::empty(
        {residual.size(0), residual.size(1), hidden}, residual.options());
    hc_prepare_kernel<<<static_cast<int>(rows), 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()), previous_pre.data_ptr<float>(),
        norm.data_ptr<float>(), normalized.data_ptr<float>(),
        reinterpret_cast<half*>(branch.data_ptr<mfq_half>()), static_cast<int>(hidden),
        static_cast<float>(epsilon));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {normalized, branch};
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    require_input(input, 32, "deepseek_v41_mxfp8_e4m3_sim");
    auto output = mfq_tensor_backend::empty_like(input);
    mxfp8_sim_kernel<<<
        static_cast<int>(input.numel() / 32), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    require_input(input, 16, "deepseek_v41_mxfp4_e4m3_scale_sim");
    auto output = mfq_tensor_backend::empty_like(input);
    mxfp4_e4m3_scale_sim_kernel<<<
        static_cast<int>(input.numel() / 16), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
