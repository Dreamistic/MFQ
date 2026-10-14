"""Retained decode graphs must survive reset, prefix restore and eager prefill."""
import asyncio
import os
from contextlib import suppress

import numpy as np
import pytest

from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import mtp_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("allocation", ["preallocated", "dynamic"])
def test_conversation_graph_reuse(tmp_path, allocation):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = mtp_fixture("qwen", 1)
    for name, value in list(weights.items()):
        if name.endswith(".experts.gate_up.weight"):
            for role, matrix in zip(("gate", "up"), np.split(value, 2, axis=1)):
                weights[name.replace("gate_up", role)] = matrix.copy()
            del weights[name]
    for name, value in list(weights.items()):
        if ".experts." in name:
            experts, output, width = value.shape
            packed = quantize(value.reshape(experts*output, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
        elif ".ngram.shard." in name:
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / "conversation.mfq"
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(reuse):
        log_path = tmp_path / f"worker-{reuse}.log"
        env = dict(os.environ, MFQ_QWEN4_REUSE_GRAPHS=str(int(reuse)),
            MFQ_KV_PLACEMENT="vram", MFQ_KV_ALLOCATION=allocation,
            MFQ_KV_TURBOQUANT_BITS="0", MFQ_MOE_PIPELINE="1",
            MFQ_MOE_ROUTER_LOOKAHEAD="0", MFQ_MOE_TWO_STAGE_FFN="1",
            MFQ_MOE_PRELOAD_ALL="1", MFQ_MOE_PREFILL_LAYER="0",
            MFQ_MOE_PREFILL_GRAPH="0", MFQ_MOE_HOST_CACHE_BYTES="1048576",
            MFQ_RUNTIME_DISABLE_PREFIX_CACHE="0", MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES="67108864",
            MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES="0", MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES="0",
            MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS="16",
            MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / f"prefix-{reuse}"))
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128",
                "--moe-gpu-cache-gb", "0.01", "--yarn-context-extension",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await client.status()
                initial_log = log_path.read_text(errors="replace")
                if reuse:
                    assert "runtime_qwen4_graph_warmup target_captures=6 predictor_captures=8" in initial_log
                    assert "runtime_qwen4_mtp_warmup tokens=64" in initial_log
                captures = initial_log.count("runtime_qwen4_graph_capture")
                outputs = []
                # A different suffix requires eager prefill into graph-owned recurrent
                # storage. The last repeated prompt then exercises prefix restoration.
                prompts = ("ab"*24, "ab"*24+"ba"*12, "ab"*24+"ba"*12, "ab"*24+"ba"*12)
                for index, prompt in enumerate(prompts):
                    if index == 3:
                        # Resizing expert arenas must discard all captured pointers
                        # before the next request restores its prefix.
                        await client.set_resident_memory_budget(1 << 30)
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 12,
                        "enable_mtp": True, "mtp_max_draft_tokens": 3, "stream": True}) as events:
                        chunks = [event async for event in events if event]
                    outputs.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                    await client.status()
                    if reuse and index < 3:
                        assert log_path.read_text(errors="replace").count("runtime_qwen4_graph_capture") == captures
                assert outputs[1] == outputs[2] == outputs[3]
                status = await client.status()
                assert status["prefix_cache_hit_tokens"] > 0
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    expected = asyncio.run(run(False))
    assert asyncio.run(run(True)) == expected
