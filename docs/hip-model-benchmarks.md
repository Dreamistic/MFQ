# HIP model benchmarks

Model-level throughput for the HIP runtime on a single MI300X (gfx942),
comparing full-precision (BF16) checkpoints with their MFQ-quantized
(S4-M) builds of the same model. Raw results:
[`bench/hip-model-bench-20261017.json`](../bench/hip-model-bench-20261017.json).

## Method

`bench/model_bench.py` drives `mfq-runtime --transport stdio` with the
native benchmark protocol: synthetic prompts sized to an exact token
count, prefix cache disabled per request, greedy sampling, and 256
decoded tokens per run. Each cell reports the median of 3 runs after 1
warmup run, batch 1, Release build. All runs set
`MFQ_DISABLE_MMA_ATTENTION=1`, which the HIP backend still requires for
prompts longer than 32 tokens (see the
[HIP validation report](hip-native-runtime-validation.md)).

```sh
python bench/model_bench.py model-bf16.mfq model-s4m.mfq \
  --runtime build/cpp_runtime/mfq-runtime \
  --prefill-lengths 32,512,2048,8192 --decode-tokens 256 \
  --env MFQ_DISABLE_MMA_ATTENTION=1 --output results.json
```

## Results (2026-10-17, MFQ master `cf78104f`)

| Model | Precision | Prompt | Decode tok/s | Prefill tok/s | TTFT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | BF16 | 32 | 265.3 | 2472.6 | 13 |
| Qwen3.5-0.8B | BF16 | 512 | 260.0 | 14267.4 | 37 |
| Qwen3.5-0.8B | BF16 | 2048 | 248.5 | 16457.9 | 125 |
| Qwen3.5-0.8B | BF16 | 8192 | 142.7 | 6559.2 | 1250 |
| Qwen3.5-0.8B | S4-M | 32 | 250.9 | 2346.6 | 14 |
| Qwen3.5-0.8B | S4-M | 512 | 246.0 | 14152.8 | 37 |
| Qwen3.5-0.8B | S4-M | 2048 | 235.6 | 16364.5 | 126 |
| Qwen3.5-0.8B | S4-M | 8192 | 138.4 | 6532.5 | 1255 |
| Qwen3.5-4B | BF16 | 32 | 134.6 | 1025.6 | 32 |
| Qwen3.5-4B | BF16 | 512 | 132.2 | 6945.7 | 74 |
| Qwen3.5-4B | BF16 | 2048 | 128.1 | 8432.5 | 244 |
| Qwen3.5-4B | BF16 | 8192 | 84.7 | 2716.8 | 3016 |
| Qwen3.5-4B | S4-M | 32 | 125.8 | 943.9 | 35 |
| Qwen3.5-4B | S4-M | 512 | 123.9 | 6401.3 | 81 |
| Qwen3.5-4B | S4-M | 2048 | 120.2 | 8171.7 | 251 |
| Qwen3.5-4B | S4-M | 8192 | 81.2 | 2740.4 | 2991 |
| Qwen3.5-9B | S4-M | 32 | 96.2 | 811.9 | 40 |
| Qwen3.5-9B | S4-M | 512 | 95.1 | 5430.5 | 95 |
| Qwen3.5-9B | S4-M | 2048 | 92.8 | 7083.7 | 290 |
| Qwen3.5-9B | S4-M | 8192 | 67.7 | 2586.5 | 3168 |

## BF16 versus S4-M

| Model | Prompt | Decode delta | Prefill delta |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 2048 | -5.2% | -0.6% |
| Qwen3.5-4B | 2048 | -6.2% | -3.1% |
| Qwen3.5-4B | 8192 | -4.1% | +0.9% |

Findings:

- S4-M decode still trails BF16, by 5-7% at 32-2048 token prompts and
  3-4% at 8192 tokens. The gap narrowed from the 15-35% range measured
  on 2026-10-12; engine-level decode improvements merged since then help
  both precisions. The remaining gap still comes from the scalar
  `__dp4a` fallback in the quantized GEMM. A hardware `v_dot4` path or
  an MFMA quantized GEMM is still required before quantized decode
  overtakes BF16.
- Prefill throughput is effectively equal between precisions. S4-M reads
  less weight data, which offsets the scalar dot-product cost.
- Prefill peaks at the 2048-token cohort for every model and drops at
  8192 tokens (quadratic attention cost); decode slows at long context
  from KV-cache growth.
- Versus the last pre-sync measurements (2026-10-12 handover), decode
  improved by 1.4-1.6x with no HIP-specific changes: 0.8B BF16 166 to
  265, 4B BF16 96 to 135, 4B S4-M 81 to 120, and 9B S4-M 64 to 93
  tok/s at the 2048-token cohort. The gains come from shared engine
  work merged to master through 2026-10-16.

Quality (KL divergence to the BF16 reference) is measured separately by
`mfq-eval`; this document covers throughput only. The Flash-Next MoE
build is excluded because generation still fails with the known logits
dtype bug, and Qwen3.5-35B-A3B remains blocked on canonical tensor
mapping.
