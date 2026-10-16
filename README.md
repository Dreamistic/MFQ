<div align="center">

# MFQ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/figures/tylogi-ai-lab-lockup-dark.svg">
  <img src="./docs/figures/tylogi-ai-lab-lockup-light.svg" alt="Tylogi AI Lab" width="420">
</picture>

### Next-generation quantization & inference infrastructure

**Every Bit. Maximum Fidelity.**

MFQ combines neural-network-aware SQ and VQ formats designed to approach the
practical rate-distortion frontier, high-quality, fine-grained mixed-precision
calibration and quantization, and efficient CUDA and Metal inference.
ROCm (HIP) support is in development. The goal is
to deliver the best model quality possible on existing hardware within its
memory, storage, and latency limits.

<p>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 license">
  <img src="https://img.shields.io/badge/runtime-C%2B%2B-black" alt="C++ runtime">
  <img src="https://img.shields.io/badge/backends-CUDA%20%7C%20Metal%20%7C%20ROCm%20%28in%20development%29-6b57ff" alt="CUDA and Metal; ROCm in development">
</p>

<p>
  <strong>English</strong> · <a href="./README.zh-CN.md">中文</a>
</p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="./docs/README.md">Documentation</a>
</p>

</div>

## Quick start

### Use a prebuilt package

Download and load models, configure serving and memory budgets, inspect
quantization, and run benchmarks from one desktop app. Applications and agents
can connect through OpenAI- and Anthropic-compatible APIs.

<p align="center">
  <img src="./docs/figures/mfq-studio-overview-en.png" alt="MFQ Studio: local model serving, live performance and resource overview" width="1100">
</p>

Download **MFQ Studio** from [Releases](https://github.com/Tylogi/MFQ/releases).
The package includes the desktop app, MFQ Server, and native runtime; no source
build is needed. Download a model from **Downloads**, then load it from
**Models**, or open an existing local `.mfq` checkpoint.

Prebuilt C++ workers can also be supplied directly to the server:

```bash
uv run mfq serve \
  --running-executable /path/to/mfq-decode-metal \
  --model /models/model.mfq
```

### Build from source

MFQ requires Git, [uv](https://docs.astral.sh/uv/), CMake 3.26+, and a
C++ toolchain. Use CUDA 12+ on NVIDIA systems or Metal on Apple silicon.
ROCm/HIP support is in development; experimental AMD builds require ROCm 6+.

```bash
git clone https://github.com/Tylogi/MFQ.git MFQ
cd MFQ

# NVIDIA / CUDA
uv sync
uv run mfq build --backend cuda

# AMD / ROCm (HIP, in development)
uv sync
uv run mfq build --backend hip

# Apple silicon / Metal
uv sync --extra metal
uv run mfq build --backend metal
```

Start an empty local server and open <http://127.0.0.1:8090/>:

```bash
uv run mfq serve
```

Detailed requirements and custom CMake options are in the
[`mfq build` guide](./docs/cli/build.md).

## Quantize and run a model

Use the MFQ **`S4-M` mixed-precision preset**. It keeps sensitive tensors at
higher precision and leaves Vision and MTP components at source precision by
default.

```bash
# Quantization dependencies
uv sync --extra train

# Hugging Face safetensors -> self-contained MFQ
uv run mfq quantize \
  /models/Qwen3.8-27B \
  /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --preset S4-M \
  --backend auto
```

Load and serve the result:

```bash
uv sync                                # add --extra metal on Apple silicon
uv run mfq build --backend auto
uv run mfq serve \
  --model /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --context-size 32768
```

MFQ can also quantize full-precision MFQ and GGUF sources, consume per-tensor or
Expert-Wise precision maps and activation importance matrices, and write
sharded output. See
[`mfq quantize`](./docs/cli/quantize.md) and
[`mfq calibrate`](./docs/cli/calibrate.md).

## Features

### Quantization

- **Fine-Grained Quantization.** Assign precision
  to individual experts, projections, chunks, or neurons within unified MFQ
  formats, retaining higher precision where the model needs it most.

- **High-Quality Calibrated Models & Rapid Architecture Support.** Download
  ready-to-run mixed-precision models, with quantization and inference support
  for new architectures added regularly.

Scalar NINT and vector NVQ formats can coexist within the same layer, with
independent choices for each expert's Gate, Up and Down projections. The
heatmaps below use **Qwen3.8-Flash-Next V3-XXS** as an example: the overall
expert average, followed by Gate, Up and Down, on a shared color scale.

<p align="center">
  <img src="./docs/figures/qwen38-flash-next-expert-precision.png" alt="Qwen3.8-Flash-Next V3-XXS: average expert precision and separate Gate, Up and Down heatmaps" width="1100">
</p>

Our calibration method is open source; see
[Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](https://arxiv.org/abs/2610.20353).

### Inference

- **Resource-Constrained Inference.** CUDA combines VRAM, system RAM and SSD
  to run models beyond GPU memory capacity. Metal supports SSD Streaming for
  MoE experts and PLE tables, loading weights on demand to reduce unified-memory
  residency.

- **Production Serving.** Continuous Batching, LRU-managed multi-model loading,
  and tiered RAM/SSD Prefix and KV caches serve applications and agents through
  OpenAI- and Anthropic-compatible APIs.

- **High-Performance CUDA & Metal Backends.** Optimized packed-weight,
  attention, cache, and I/O kernels for CUDA and Metal. ROCm/HIP support is
  in development; optimized CPU backends are planned.

## Models

Published `.mfq` models are available from
[Hugging Face](https://huggingface.co/Tylogi) and
[ModelScope](https://www.modelscope.cn/profile/Tylogi).

| Model | Published precision families | Download |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `V1-S`, `V1-M`, `S2-L`, `V3-XXS`, `S3-L`, `V4-XS`, `S4-L`, `S5-L` | [Hugging Face](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ) · [ModelScope](https://www.modelscope.cn/models/Tylogi/Qwen3.8-Flash-Next-EW-MFQ) |
| DeepSeek-V4-Flash-0731 | Expert-Wise `V2` tiers | [Hugging Face](https://huggingface.co/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) · [ModelScope](https://www.modelscope.cn/models/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) |
| Qwen3.8-27B | `V1`–`V4`, `S4`–`S6` tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/Qwen3.8-27B-MFQ) |
| MiniCPM-o 4.5 | `S4`–`S8` multimodal tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/MiniCPM-o-4_5-MFQ) |

Supported architecture groups:

| Architecture group | Backends |
| --- | --- |
| Qwen3.5–3.8 | CUDA C++, Metal C++, HIP C++ (dense models; in development) |
| Qwen Flash-Next / Qwen4-style | CUDA C++, Metal C++ |
| DeepSeek-V4-Flash Series | CUDA C++, Metal C++ |
| MiniCPM-o 4.5 | CUDA C++, Metal C++ |
| GLM5–5.3 | CUDA C++; Metal reference runtime |
| Gemma 4 | CUDA C++; Metal reference runtime |

See the [runtime support matrix](./docs/runtime-support.md) for details.

## Benchmarks

### Qwen3.8-Flash-Next: local inference speed

#### Apple

| Tier | RAM Usage | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | 43.36 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.5 tok/s | 71.6 tok/s | 101.2 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | 47.41 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.9 tok/s | 71.0 tok/s | 97.9 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | 50.33 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,527.6 tok/s | 70.8 tok/s | 105.4 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | 57.95 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,565.6 tok/s | 68.1 tok/s | 98.2 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | 61.48 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,540.1 tok/s | 67.7 tok/s | 96.7 tok/s |
| [S4-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S4-L) | 78.25 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,481.7 tok/s | 63.6 tok/s | 93.6 tok/s |

#### CUDA

| Tier | RAM Usage | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | ≈41.04 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,779.7 tok/s | 62.9 tok/s | 87.0 tok/s |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | ≈42.90 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,787.2 tok/s | 63.4 tok/s | 77.5 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | ≈46.93 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,707.8 tok/s | 58.8 tok/s | 77.8 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | ≈49.81 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,626.2 tok/s | 51.7 tok/s | 61.0 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | ≈57.28 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,408.5 tok/s | 39.4 tok/s | 42.1 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | ≈60.72 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,317.1 tok/s | 36.0 tok/s | 36.2 tok/s |

The PLE table stays on SSD; it does not need to occupy resident memory.
For example, V1-M downloads 72.44 GiB, including 29.54 GiB of PLE, while its
resident weights occupy 43.36 GiB.

### Qwen3.8-Flash-Next: model size vs. KLD and Top1

WikiText-2 evaluation: lower KLD is better; higher Top1 agreement with the
reference model is better.

<p align="center">
  <img src="./docs/figures/qwen38-flash-next-quality.png" alt="Qwen3.8-Flash-Next: MFQ, Unsloth UD and GSQ-RCO model size versus KLD and Top1 agreement" width="1100">
</p>

| MFQ Tier | Weight size (GB) | KLD ↓ | Top1 agreement ↑ |
| --- | ---: | ---: | ---: |
| `V1-S` | 72.536 | 0.22157 | 83.77% |
| `V1-M` | 74.529 | 0.19689 | 84.72% |
| `S2-L` | 78.859 | 0.14630 | 86.54% |
| `V3-XXS` | 81.952 | 0.12436 | 87.62% |
| `S3-L` | 89.976 | 0.08698 | 89.55% |
| `V4-XS` | 93.672 | 0.07524 | 90.36% |

At approximately 74.5 GB, **V1-M reduces KLD by 37.4% and improves Top1
agreement by 5.03 percentage points** over Unsloth UD IQ1_M.

Weight sizes in this comparison exclude the vision encoder and MTP head;
GB = 10⁹ bytes. See the [model card](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ)
for the full comparison and tier selection guide.

## Documentation

- [Documentation index](./docs/README.md)
- [Build the C++ runtime](./docs/cli/build.md)
- [Quantize a model](./docs/cli/quantize.md)
- [Run MFQ Server](./docs/cli/serve.md)
- [HTTP API](./docs/api/http.md)
- [WebSocket and realtime API](./docs/api/websocket.md)
- [Runtime support](./docs/runtime-support.md)
- [MiniCPM-o 4.5 multimodal runtime](./docs/minicpmo45.md)
- [Contributing and architecture rules](./CONTRIBUTING.md)

## Acknowledgements

MFQ is deeply inspired by and has learned from
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[oMLX](https://github.com/jundot/omlx),
[MLX](https://github.com/ml-explore/mlx),
[PyTorch](https://github.com/pytorch/pytorch),
[Transformers](https://github.com/huggingface/transformers), and
[Unsloth](https://github.com/unslothai/unsloth).

MFQ is licensed under the [Apache License 2.0](./LICENSE).
