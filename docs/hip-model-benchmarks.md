# HIP model benchmarks

Model-level throughput for the HIP runtime on a single MI300X (gfx942),
comparing true full-precision (BF16) MFQ conversions with their MFQ-quantized
(S4-M) builds across the dense Qwen3.5 size ladder and Qwen3.8-27B. Raw
results: [`bench/hip-model-bench-20261017-bf16.json`](../bench/hip-model-bench-20261017-bf16.json)
and [`bench/hip-model-bench-20261017-quantized.json`](../bench/hip-model-bench-20261017-quantized.json).

## Method

`bench/model_bench.py` drives `mfq-runtime --transport stdio` with the
native benchmark protocol: synthetic prompts sized to an exact token
count, prefix cache disabled per request, greedy sampling, and 256
decoded tokens per run. Each cell is the median of 3 runs after 1
warmup run, batch 1, Release build. All runs set
`MFQ_DISABLE_MMA_ATTENTION=1`, which the HIP backend still requires for
prompts longer than 32 tokens (see the
[HIP validation report](hip-native-runtime-validation.md)).

```sh
python bench/model_bench.py model-fp.mfq model-s4m.mfq \
  --runtime build/cpp_runtime/mfq-runtime \
  --prefill-lengths 32,512,2048,8192 --decode-tokens 256 \
  --env MFQ_DISABLE_MMA_ATTENTION=1 --output results.json
```

Full-precision references are `mfq quantize <hf-dir> model-fp.mfq
--full-precision --tokenizer <tokenizer.gguf>`. The `--tokenizer` asset
is required: without it the file loads no tokenizer and the runtime
exits with "model runtime requires model config and tokenizer GGUF".

## Reference-file correction

The earlier "BF16" reference files (`qwen35-0.8b.mfq`,
`qwen35-4b-bf16.mfq`) were default-recipe NINT conversions, not
full-precision files: the 4B file stores 2.21 GiB NINT and 0.85 GiB
BF16. Earlier tables that compared those files against S4-M measured
NINT against NINT. This document compares only true full-precision
files against S4-M. Weight sizes below are the stored tensor bytes.

## Decode and prefill (2026-10-17, master `cf78104f`, 2048-token cohort)

| Model | Weights BF16/S4-M | Decode BF16 | Decode S4-M | Speedup | Prefill BF16 | Prefill S4-M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.6 / 0.8 GiB | 165.0 | 236.2 | 1.43x | 16744 | 16348 |
| Qwen3.5-2B | 4.2 / 1.9 GiB | 172.4 | 192.4 | 1.12x | 15006 | 14522 |
| Qwen3.5-4B | 8.7 / 3.2 GiB | 118.1 | 120.1 | 1.02x | 8654 | 8451 |
| Qwen3.5-9B | 18.0 / 6.8 GiB | 71.6 | 92.8 | 1.30x | 7366 | 7069 |
| Qwen3.5-27B | 51.8 / 17.5 GiB | 22.0 | 36.2 | 1.64x | 2769 | 2517 |
| Qwen3.8-27B | 51.8 / 17.5 GiB | 22.1 | 36.2 | 1.64x | 2782 | 2537 |

Decode speedup by cohort: at 32 tokens 1.02x (4B) to 1.67x (Qwen3.8-27B);
at 8192 tokens 1.01x to 1.50x. S4-M decode wins at every size; the gain
grows with model weight. Prefill is 1-9% slower for S4-M at 512-2048
tokens and up to 2.5x slower at 32 tokens (998 vs 1684 tok/s on 4B),
which mirrors the launch-bound short-prompt regime where the dequant
fallback cannot amortize (see diagnosis).

## Diagnosis: where quantized speed is lost

An op-level profile (`mfq-diagnostics --profile --gen 65`, GPU event
timings, 4B models) locates the remaining gap:

- **Decode is not bandwidth-bound on the quantized path.** Effective
  weight traffic at the 2048 cohort is 1.0-1.3 TB/s for BF16 versus
  0.4-0.65 TB/s for S4-M (MI300X peak is 5.3 TB/s). The S4-M kernels
  convert only 35-55% of the theoretical 2.6-3.0x weight reduction into
  speed; the BF16 GEMV path reaches roughly twice the bandwidth
  efficiency of the NINT GEMV kernels.
- **Mixed-precision width costs real time.** S4-M stores the GDN qkv
  projection wider than the uniform default (16.5 vs 11.3 MiB per
  layer on 4B). The resulting GEMV variant costs 2.86 ms per token
  inside `linear.in_proj` versus 0.85 ms for the uniform-width file,
  about 2 ms per token of the 8.3 ms budget.
- **Prefill above 8 rows dequantizes whole matrices.** `nint_matmul`
  (cpp_runtime/backends/hip/ops/nint.cpp) uses the fused workspace GEMV
  only for M <= 8; larger cohorts dequantize the full weight to BF16
  and call a dense GEMM. In the profiled window the dequant kernel
  alone cost 11.3 ms against 6.5 ms for the GEMM it feeds, and
  `linear.in_proj` ran 6x slower than the BF16 equivalent (0.274 vs
  0.045 ms per call).
- **Eager totals confirm the ordering.** Without decode graphs, per
  token costs were 16.4 ms (uniform NINT), 19.8 ms (S4-M), and 23.2 ms
  (BF16); the graph path compresses all three but keeps the order.

Fix directions, in expected impact order: vectorized bit-unpacking with
wave-resident scales in the fused GEMV; a hardware `v_dot4`/MFMA integer
dot (the known `__dp4a` scalar fallback); routing the S4-M wide tensors
through the aligned-q8 fast kernel where quality allows; extending the
fused path beyond M <= 8 or caching dequantized hot tensors for
prefill-sized cohorts.

Quality (KL divergence to the BF16 reference) is measured separately by
`mfq-eval`; this document covers throughput only. The Flash-Next MoE
build is excluded because generation still fails with the known logits
dtype bug, and Qwen3.5-35B-A3B remains blocked on canonical tensor
mapping.
