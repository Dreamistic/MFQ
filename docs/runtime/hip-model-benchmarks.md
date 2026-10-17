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
