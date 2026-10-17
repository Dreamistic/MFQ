# HIP model benchmarks

Model-level throughput for the HIP runtime on a single MI300X (gfx942),
comparing true full-precision (BF16) MFQ conversions with their MFQ-quantized
(S4-M) builds across the dense Qwen3.5 size ladder and Qwen3.8-27B, measuring
speculative MTP decode, and referencing an independent llama.cpp run on the
same GPU. Raw results:
[`bench/hip-model-bench-20261017-bf16.json`](../../bench/hip-model-bench-20261017-bf16.json),
[`bench/hip-model-bench-20261017-quantized.json`](../../bench/hip-model-bench-20261017-quantized.json),
[`bench/hip-model-bench-20261017-mtp.json`](../../bench/hip-model-bench-20261017-mtp.json),
and [`bench/hip-llama-bench-20261017.txt`](../../bench/hip-llama-bench-20261017.txt).

## Method

`bench/model_bench.py` drives `mfq-runtime --transport stdio` with the
native benchmark protocol: synthetic prompts sized to an exact token
count, prefix cache disabled per request, greedy sampling, and 256
decoded tokens per run. Each cell is the median of 3 runs after 1
warmup run, batch 1, Release build. All runs set
`MFQ_DISABLE_MMA_ATTENTION=1`, which the HIP backend still requires for
prompts longer than 32 tokens (see the
[HIP validation report](hip-validation.md)).

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
timings, 4B models) plus kernel-source analysis and A/B runs locate the
gap:

- **The packed NINT GEMV kernel is the bottleneck, and its cost scales
  with weight bytes at a fixed rate.** Normalizing end-to-end decode by
  stored bytes, the uniform-NINT and S4-M files both stream at about
  0.41 TB/s on 4B and 0.63 TB/s on 27B, independent of bit width. On
  the same GPU the MFQ BF16 path sustains 1.14 TB/s and llama.cpp
  sustains 1.2-1.25 TB/s at 27B. Every other part of the stack is
  competitive; the fused decode kernel converts weight bytes to
  answers at roughly half the demonstrated streaming rate.
- **The scalar `__dp4a` fallback is no longer active.** A probe kernel
  compiled with the current toolchain shows `__gfx942__` defined and
  `__has_builtin(__builtin_amdgcn_sdot4)` true, emitting `v_dot4`. The
  `mfq_hip_compat.h` guard takes the hardware path in current builds;
  the earlier "no hardware dot" finding predates the newer compiler.
- **The per-chunk float epilogue is worth 3-4%.** The
  integer-accumulate path (`nint_group_partial`, previously opt-in via
  `MFQ_NINT_GROUP_DOT=1`) measured +3.4% decode on 4B S4-M (121.2 to
  125.3 tok/s) and +4.4% on 27B (36.4 to 38.0) before further kernel
  work; see the improvement section below.
- **Prefill above 8 rows dequantizes whole matrices.** `nint_matmul`
  (cpp_runtime/backends/hip/ops/nint.cpp) uses the fused workspace GEMV
  only for M <= 8; larger cohorts dequantize the full weight to BF16
  and call a dense GEMM. In the profiled window the dequant kernel
  alone cost 11.3 ms against 6.5 ms for the GEMM it feeds.

What remains is the kernel's per-byte instruction and access pattern:
one output row per 32-lane subwarp, 4-8 byte weight loads per chunk
followed by shift/mask unpacking, and three scattered per-group scale
loads (subgroup scale, minimum, activation scale) that are separate
from the weight stream. llama.cpp's Q4_K_M reaches 1.2 TB/s with an
inline block layout: scales stored next to the weights they scale and
32-byte aligned blocks, so one wide load serves both.

## Kernel improvement (2026-10-17)

The integer group-accumulation path (`nint_group_partial`) is now the
default for single-row decode (`MFQ_NINT_GROUP_DOT=0` still opts out),
and its weight loads use a sliding dword window: adjacent chunks reuse
the previously loaded words, so each window advance costs one load
instead of two per chunk. The unpack arithmetic is unchanged, so the
grouped path stays within the same tolerance of the plain path as
before (`mfq-hip-nint-q8-test`, 3 consecutive runs, identical bounds).

| Model | Plain path tok/s | New default tok/s | Gain |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B S4-M | 120.3 | 129.5 | +7.6% |
| Qwen3.5-9B S4-M | 92.9 | 98.3 | +5.8% |
| Qwen3.5-27B S4-M | 36.2 | 40.4 | +11.6% |

Measurements are medians of 3 runs at the 2048-token cohort; prefill is
unchanged. The remaining gap to the 1.2 TB/s streaming rate needs the
larger layout work below.

Fix directions, in expected impact order: widen weight loads to 16
bytes per iteration and co-locate group scales with the weight stream;
process more than one activation row per wave to amortize weight
traffic once MTP batches arrive; extend the fused path beyond M <= 8 or
cache dequantized hot tensors for prefill-sized cohorts.

## Speculative MTP decode (2026-10-17, 2048-token cohort)

`bench/model_bench.py --mtp` repeats the decode matrix with
`enable_mtp` and records drafted and accepted tokens. Raw results:
[`bench/hip-model-bench-20261017-mtp.json`](../../bench/hip-model-bench-20261017-mtp.json).

| Model | Precision | Plain tok/s | MTP tok/s | Ratio | Acceptance | Drafted/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | BF16 | 165.0 | 80.0 | 0.48x | 41% | 0.15 |
| Qwen3.5-0.8B | S4-M | 236.0 | 119.3 | 0.51x | 20% | 0.06 |
| Qwen3.5-2B | BF16 | 172.5 | 83.6 | 0.48x | 63% | 0.14 |
| Qwen3.5-2B | S4-M | 193.2 | 108.0 | 0.56x | 40% | 0.06 |
| Qwen3.5-4B | BF16 | 117.6 | 71.4 | 0.61x | 92% | 0.77 |
| Qwen3.5-4B | S4-M | 120.3 | 109.1 | 0.91x | 90% | 0.79 |
| Qwen3.5-9B | BF16 | 71.5 | 60.0 | 0.84x | 100% | 0.73 |
| Qwen3.5-9B | S4-M | 91.9 | 93.9 | 1.02x | 100% | 0.73 |
| Qwen3.5-27B | BF16 | 22.0 | 15.7 | 0.71x | 64% | 0.33 |
| Qwen3.5-27B | S4-M | 36.3 | 26.5 | 0.73x | 56% | 0.39 |
| Qwen3.8-27B | BF16 | 22.1 | 20.2 | 0.92x | 90% | 0.77 |
| Qwen3.8-27B | S4-M | 36.2 | 38.7 | 1.07x | 90% | 0.78 |

MTP is a net loss on HIP today for every model below 9B and for the
BF16 27B builds; only Qwen3.5-9B and Qwen3.8-27B S4-M reach break-even
or a small gain (up to 1.07x). Two signals point at per-step MTP
overhead rather than the predictor quality: small models draft rarely
(0.06-0.15 drafted tokens per output token) yet still lose half their
throughput, and acceptance itself is healthy (90-100%) where drafting
engages. The MTP stall and workspace optimizations that landed for the
CUDA backend have no HIP equivalent yet; porting them is the first fix
to try before MTP is useful here.

## llama.cpp reference on the same GPU

An independent reference was measured with llama.cpp (`f74e2fa`, HIP
backend, gfx942, same MI300X host, `llama-bench -p 32,2048 -n 256 -r 3
-ngl 99`). Models were converted from the same HF snapshots
(`convert_hf_to_gguf.py --outtype f16`) and quantized to Q4_K_M with
`llama-quantize`. Numbers are means of 3 repeats; llama-bench `tg` runs
at its default context, so compare against the MFQ 32-token cohort for
the closest KV size.

| Model | llama F16 tg | MFQ BF16 | llama Q4_K_M tg | MFQ S4-M |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 406.8 | 173.2 | 414.4 | 251.8 |
| Qwen3.5-2B | 290.7 | 181.6 | 375.6 | 203.3 |
| Qwen3.5-4B | 180.3 | 123.6 | 242.6 | 125.7 |
| Qwen3.5-9B | 122.9 | 73.6 | 193.8 | 96.3 |
| Qwen3.5-27B | 40.7 | 22.4 | 77.1 | 37.2 |
| Qwen3.8-27B | 40.6 | 22.4 | 76.9 | 37.3 |

Prefill is competitive: at the 2048 cohort MFQ BF16 reaches 2769-16744
tok/s against llama F16 pp2048 of 2523-19731 tok/s (within 10% at
every size). Decode is where the HIP backend trails: llama.cpp F16
decodes 1.5-1.8x faster than the MFQ BF16 path, and llama.cpp Q4_K_M
decodes 1.6-2.1x faster than MFQ S4-M. llama.cpp also converts more of
its quantization into speed (1.9x Q4_K_M over F16 at 9B/27B versus
1.3-1.64x for S4-M over BF16), which independently confirms the
bandwidth-efficiency diagnosis above: the gap is in the quantized GEMV
kernels, not in the format.

Quality (KL divergence to the BF16 reference) is measured separately by
`mfq-eval`; this document covers throughput only. The Flash-Next MoE
build is excluded because generation still fails with the known logits
dtype bug, and Qwen3.5-35B-A3B remains blocked on canonical tensor
mapping.
