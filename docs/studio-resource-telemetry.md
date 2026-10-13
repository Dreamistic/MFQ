# Studio resource tiers

The overview displays an OpenAI SDK base URL ending in `/v1`. This is a display
and clipboard URL; Studio's management API base URL is unchanged.

`GET /api/v1/runtime/instances` includes an optional `memory` breakdown per
instance. Studio aggregates ready/busy instances, not just the selected model:

| Tier | Measurement |
| --- | --- |
| Resident experts and dense weights | `resident_weight_bytes`, model weight allocations |
| Resident KV and prefix cache | `kv_bytes`, live context state plus resident prefix bytes; context groups and resident prefix blocks |
| SSD-streamed experts | `ssd_expert_bytes`, backing record payloads when streaming is enabled |
| SSD-streamed PLE tables | `ssd_ple_bytes`, backing row-table record payloads when streaming is enabled |

SSD payloads are not added to resident memory. Cumulative bytes read, allocator
reuse caches and process RSS are not substituted for these measurements.
Missing telemetry is `null`, not zero. Busy workers can retain the last reported
breakdown when their nonblocking telemetry lock is unavailable.

The native Metal worker aggregates counters registered by cache and storage
components, independently of model names or backbone-specific runner methods.
MFE format/geometry probes do not register SSD backing. A record enters that
count only when a layer switches to streaming or actually pages its experts;
only those active records are counted after a partial weight offload. Their
cached RAM pages remain in dynamic weight residency. SSD backing denotes file
payloads, not an additional RAM allocation, and cache hits do not duplicate it.
New backbones reusing these components inherit telemetry; a new storage/cache
component registers its own resource lifetime once. Refresh does not evaluate
devices, copy arrays or read files. Static weight residency is sampled
after load staging is released, before generation; dynamic MFE weight residency
is added separately. Preallocated expert arenas belong to weight residency.
Live cache bytes include attention K/V, index caches, GDN recurrent state and
PLE convolution state, including predictor caches. They describe retained cache
storage capacity, not just visible token rows. Temporary speculative work is
not included. Disk-only prefix blocks/contexts are excluded from resident counts.
Workers without these counters explicitly show unavailable breakdowns. Partial
aggregates retain known values as lower bounds and identify missing models.

Each model gets one color shared by all four tiers. Registration follows load
order: existing accent blue, then red, green, yellow and additional colors.
Unloading a model does not recolor the remaining models during the view's
lifetime. The legend is present even for a single model.

The resident-weight track uses the detected usable inference memory limit
(`runtime_memory_effective_budget_bytes`, falling back to the configured runtime
budget on older servers). Its right-hand label displays numeric used / limit
values. The KV/prefix track uses that same limit minus the resident weights of
all ready/busy instances, clamped to zero. Resident segments represent utilization
of their respective limits, not shares normalized to a full bar. On overcommit,
the bar is full and retains all model colors; the numeric limit is not increased.
SSD segment widths remain each model's share of that tier's reported bytes.
Missing measurements are excluded and marked as unavailable. An unknown memory
limit, or unknown weight residency when calculating KV capacity, is displayed as
`--` and does not produce a misleading full resident bar.

Loading bars appear in the overview, loaded-model list and local checkpoints.
Native workers report unique tensor records whose sources have been prepared
(read, mapped or parsed for streaming), followed by runtime finalization. These
are preparation steps, not elapsed-time estimates or physical SSD bytes read.
Aliases share one step and metadata assets do not count. Unsupported workers
show indeterminate progress; only a ready runtime completes the bar.

CUDA resources use physical GPU IDs. `runtime/status.gpu_devices` contains
live device capacity and free space; `instances[].memory.devices` reports each
worker's weights, buffers, KV and prefix allocations on those devices. Worker
ordinals are translated through the load request's `CUDA_VISIBLE_DEVICES` order.
Studio displays one VRAM section per GPU, including idle devices, and one shared
RAM/SSD section. A worker's aggregate allocation is never duplicated on every
selected GPU. Unified CUDA hardware retains the unified-memory layout.

The model load policy selects one GPU for each worker. Separate workers can
use different GPUs, with budgets and allocations tracked per device. Loading
one model across GPUs, including layer placement and tensor parallelism, is
deferred.

Single-device CUDA workers accept online resident and weight budgets while
idle. Graphs and transient workspaces are retired before resizing compressed
expert arenas; evicted experts retain RAM copies. The minimum expert working set
and non-offloadable weights bound budget reductions. Prefix snapshots survive
successful changes. A global policy is additionally capped per physical GPU.
Changing the expert partition invalidates the previous contiguous prefill image;
prefill then uses the existing per-expert transfer path until the model reloads.
CUDA Qwen workers also accept the same YaRN context extension setting as Metal.
CUDA Qwen3.5 full attention and Qwen4 QSA support TurboQuant KV at 2, 2.5, 3,
3.5, 4, 6 and 8 bits. Half-bit settings use floor(bits) for keys and ceil(bits)
for values. Decode attention consumes packed KV directly; graph decoding, MTP,
paged batching, KV offload and SSD prefix persistence preserve the packed
representation. QSA prefill temporarily decodes one layer's KV to FP16 when
the floating workspace fits within 128 MiB, then uses the batched attention
kernels. Larger caches retain direct packed attention. Indexer caches and recurrent state retain their original
precision. Prefix compatibility includes the quantization setting. Changing
precision reloads the worker through the same Studio context setting as Metal.

CUDA validation covers all seven levels against decoded CPU attention, graph
position changes and rollback, paged KV, RAM/SSD eviction, prefix restoration
after worker restart, MTP graph replay and two simultaneous requests. Current
full-model validation uses the five-shard S2-L release of
`Tylogi/Qwen3.8-Flash-Next-EWQ-V1-MFQ`, revision
`9fb4bec16958bf7c490735fdfa82cf4d8c76cc58`; all five official SHA256 values
matched. The earlier four-shard checkpoint is excluded from these results.

The Qwen4 CUDA adapter now loads the declared grid ViT component. Single-image
input works with ordinary generation and MTP, including three-axis positions
followed by text decode positions. Video and multiple images are not supported
by this component.

MTP prompt priming uses bounded 1,024-token chunks and the combined Gate/Up
whole-layer expert path. Predictor images use their own capacity and do not
prefetch backbone images. Priming time is included in request prefill metrics.
On Windows, predictor RAM images remain pageable to preserve pinned-memory
headroom for the decode transfer workspace.

Allocation failures report resource exhaustion, reclaim pending allocations
before rejecting a request, and preserve scheduler recovery. The real-model
FP16 16K prompt that previously exceeded the configured VRAM limit now completes;
subsequent image, prefix and concurrent requests also complete.

Measured on RTX 3090 Ti (24 GiB), with a 22 GiB process VRAM budget,
10.66 GiB expert GPU cache, 32K context limit and 8,192-token target prefill
chunks. Short decode results use the second warmed request, a 27-token prompt,
greedy sampling and MTP draft limit 3. Long-prompt tests use synthetic repeated
text; these are throughput measurements, not a quality evaluation.

The full matrix below preceded the final exchange synchronization repair;
final targeted measurements follow it, with their request order recorded.

| Test | FP16 KV | 4-bit KV |
| --- | ---: | ---: |
| Short ordinary decode, tok/s | 60.43 | 54.72 |
| Short MTP decode, tok/s | 73.63 | 70.62 |
| 4K prefill, tok/s | 1700 | 1723 |
| 8K prefill, tok/s | 1967 | 1965 |
| 16K prefill, tok/s | 1726 | 1751 |
| 8K-context ordinary decode, tok/s | 42.14 | 34.43 |
| 8K-context MTP decode, tok/s | 47.56 | 43.72 |
| Two concurrent requests, aggregate decode tok/s | 79.50 | 69.25 |
| SSD prefix restore, first-token ms (1,282 cached tokens) | 681 | 678 |

The 4-bit 8K prefill rate increased from 1,000 to 1,965 tok/s. The 8K MTP
first-token latency fell from 23.27 seconds to approximately 4.6 seconds after
the predictor's expert transfer path was corrected. Per-request concurrency
decode rates are summed; prefill time is excluded from that aggregate.

Local raw records are in `workspace/turboquant-smoke/matrix-fp16-final.json`,
`matrix-tq4-final.json` and `s2l-v1-sha256.json`. The original
`vision-mtp-long` row in `matrix-tq4-final.json` accidentally repeated the short
color question and is excluded from long-output MTP verification.

The final streaming matrix (`matrix-stream-final3.json`) enables 4-bit KV with
only 20 MiB of GPU KV budget (5 MiB staging) and 1 MiB of RAM KV budget, forcing
SSD traffic. Single-image MTP produces 64 tokens with 22 speculative cycles;
4K text MTP and two concurrent 4K requests complete. Recorded SSD reads exceed
955 MB, with the GPU hot tier below 15 MiB and the RAM hot tier below 1 MiB.
Under this deliberately severe limit, 4K prefill reaches 1,163 tok/s and decode
is 4.69 tok/s; these are eviction stress results, not normal-budget speeds.

This combination exposed a residency/MTP synchronization bug: eager predictor
calls could reuse transfer state before the backbone's pending exchange was
published, and the background worker could observe a subsequently overwritten
compute event. Each exchange now captures its own completion event, and entry
to the predictor completes pending exchanges. The 64-token image request that
previously stalled now completes in 6.77 seconds under the streaming stress
configuration. Native integration covers 17 cases; the residency fixture also
checks the predictor boundary before verifying expert ownership and RAM bytes.

After the exchange synchronization fix, `matrix-check-final.json` records
54.45 tok/s ordinary decode, 69.37 tok/s MTP and 1,916 tok/s 8K prefill with
4-bit KV. The 64-token image test measures 19.26 tok/s without MTP and 19.63
tok/s with MTP (24 accepted tokens). Visual target decoding currently uses
the eager path because its position history has multiple axes. The shorter
verification sequence measures 30.58 tok/s for 8K MTP; benchmark order and
accepted-token counts are retained in the raw records rather than treating
this as interchangeable with the warmed full matrix above.

The long-context sequence also exposed a predictor prefill allocation failure
after prior 16K/8K requests: its 2,040,305,664-byte layer image could exceed the
remaining 22 GiB process budget. Whole-layer staging allocation now falls back
to the bounded expert-group ring. Combined Gate/Up shares its packed transfer
and projection in that path as well. Pageable predictor fields are accepted;
subsequent prompt chunks reuse the smaller ring. Allocation failures elsewhere
still preserve the existing recoverable resource-error behavior.

`matrix-check-long2.json` completes all six requests after that fix, with a
logged grouped fallback at predictor layer -1. Final 4-bit measurements are
1,952 tok/s for 8K prefill, 1,734 tok/s for 16K prefill, 24.53 tok/s ordinary
8K-context decode, and 27.02/38.22 tok/s for first/repeated 8K MTP (20/54
accepted tokens). MTP first-token latency is 4.71/4.69 seconds. Both prefill
lengths use 8K target compute chunks; MTP priming uses 1K chunks. These latest
long-context results supersede the earlier 43.72 tok/s MTP measurement above
for reporting the final build. The short-prompt 69.37 tok/s result does not
describe 8K-context decoding.

## Discrete-device RAM prefix tier (2026-10-13)

CUDA prefix caching now uses native VRAM snapshots, a disposable serialized RAM
tier, and optional SSD storage. RAM stores content-addressed blocks without
immediate disk writes. LRU eviction writes dirty blocks to SSD when enabled;
with SSD disabled it drops them, including blocks referenced by session bindings.
Missing prefixes are recomputed. Unified-memory cache placement is unchanged.

The shared RAM allowance defaults to 2 GiB across workers and can be changed
online with `prefix_ram_limit_bytes`; zero disables the tier. Per-model requests
can further cap it with `prefix_cache_ram_bytes`. This is a ceiling, not reserved
memory. Host pressure reduces the allowance, native monitoring reclaims during
active inference, and streamed KV allocation requests RAM before allocating its
staging buffers. Model admission releases idle workers' RAM prefixes first.
The native monitor preserves 512 MiB of host headroom and checks every 250 ms.
Windows model unload drains RAM prefixes before terminating its worker.

Overview/resource panels show RAM prefixes separately from VRAM KV. The cache
page reports all three tiers and separate VRAM/RAM/SSD hit counts. Service
settings expose the RAM allowance and SSD switch; pending SSD payloads remain
included in RAM usage until the write completes.

Validation includes partial/full LRU reclaim, pinned-block reclamation, host
pressure and allocation callbacks, SSD enable/disable, CUDA RAM/SSD restores
with TurboQuant, shared worker quotas, accounting, and UI controls. The dedicated
CUDA RAM tests and existing SSD/MTP/batching matrix pass. Windows lacks symlink
creation privileges for two existing filesystem tests; those cases remain
outside the Windows verification run.
