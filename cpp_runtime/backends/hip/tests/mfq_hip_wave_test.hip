#include "mfq_hip_compat.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Result {
    int shuffle8, shuffle16, shuffle32, down8, up16, xor32;
    float floating16;
    unsigned votes48, upper, masked;
};
__global__ void check_waves(Result* results) {
    const int thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    auto& result = results[thread];
    result.shuffle8 = __shfl_sync(0xffffffffu, thread, 1, 8);
    result.shuffle16 = __shfl_sync(0xffffffffu, thread, 2, 16);
    result.shuffle32 = __shfl_sync(0xffffffffu, thread, 3);
    result.down8 = __shfl_down_sync(0xffffffffu, thread, 1, 8);
    result.up16 = __shfl_up_sync(0xffffffffu, thread, 1, 16);
    result.xor32 = __shfl_xor_sync(0xffffffffu, thread, 4);
    result.floating16 = __shfl_sync(0xffffffffu, float(thread), 3, 16);
    result.votes48 = __ballot_sync(0xffffffffu, thread % 64 < 48);
    result.upper = __ballot_sync(0xffffffffu, thread % 64 >= 32);
    result.masked = __ballot_sync(0x55555555u, thread % 3 == 0);
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
}

int main() {
    try {
        int devices = 0;
        const auto status = hipGetDeviceCount(&devices);
        if (status == hipErrorNoDevice || status == hipErrorInsufficientDriver ||
            (status == hipSuccess && devices == 0)) return 77;
        check(status);
        std::vector<Result> results(256);
        Result* device = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&device), results.size() * sizeof(Result)));
        check_waves<<<1, dim3(8, 8, 4)>>>(device);
        check(hipGetLastError());
        check(hipMemcpy(results.data(), device, results.size() * sizeof(Result), hipMemcpyDeviceToHost));
        check(hipFree(device));
        for (int thread = 0; thread < 256; ++thread) {
            const auto& value = results[thread];
            require(value.shuffle8 == thread / 8 * 8 + 1, "width8 shuffle crossed output rows");
            require(value.shuffle16 == thread / 16 * 16 + 2, "width16 shuffle crossed output rows");
            require(value.shuffle32 == thread / 32 * 32 + 3, "default shuffle crossed logical warps");
            require(value.down8 == (thread % 8 == 7 ? thread : thread + 1), "width8 shuffle down differs");
            require(value.up16 == (thread % 16 == 0 ? thread : thread - 1), "width16 shuffle up differs");
            require(value.xor32 == (thread ^ 4), "default shuffle xor differs");
            require(value.floating16 == float(thread / 16 * 16 + 3), "floating shuffle differs");
            require(value.votes48 == ((thread / 32 & 1) ? 0xffffu : 0xffffffffu),
                "48-route ballot count differs across half waves");
            require(value.upper == ((thread / 32 & 1) ? 0xffffffffu : 0u),
                "upper half-wave predicate was lost");
            unsigned expected = 0;
            for (int lane = 0; lane < 32; ++lane)
                if ((thread / 32 * 32 + lane) % 3 == 0) expected |= 1u << lane;
            require(value.masked == (expected & 0x55555555u), "masked ballot differs");
        }
        std::cout << "HIP subwarp and half-wave semantics passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
