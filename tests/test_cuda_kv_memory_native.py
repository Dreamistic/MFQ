"""Native placement policies: capacity admission, host access and spill ordering."""
import asyncio
import os
from contextlib import suppress
from pathlib import Path

import pytest

from mfq.server.runtime.client import StdioRuntimeClient
from tests.test_cuda_flash_next_runtime import qwen_model_fixture, write_qwen_fixture
from tests.test_native_nint_ple import _worker_tokenizer


@pytest.mark.parametrize("placement,allocation,gpu_limit,ram_limit", [
    ("vram", "preallocated", 0, 0),
    ("ram", "preallocated", 0, 1 << 20),
    ("auto", "preallocated", 8192, 1 << 20),
    ("auto_ssd", "preallocated", 8192, 4096),
    ("auto", "dynamic", 0, 0),
])
def test_kv_policy_generation(tmp_path, placement, allocation, gpu_limit, ram_limit):
    executable = os.environ.get("MFQ_CUDA_RESOURCE_TEST_RUNTIME")
    if not executable:
        pytest.skip("CUDA runtime required")
    config, weights = qwen_model_fixture(ple=False)
    model = tmp_path / "tiny.mfq"
    write_qwen_fixture(model, config, weights)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(mode, limit, ram, name):
        env = dict(os.environ, MFQ_KV_PLACEMENT=mode, MFQ_KV_ALLOCATION=allocation,
            MFQ_KV_GPU_LIMIT_BYTES=str(limit), MFQ_KV_RAM_LIMIT_BYTES=str(ram),
            MFQ_KV_TURBOQUANT_BITS="0", MFQ_RUNTIME_DISABLE_PREFIX_CACHE="1",
            MFQ_RUNTIME_PREFIX_CACHE_DIR=str(tmp_path / name))
        with (tmp_path / (name + ".log")).open("wb") as log:
            process = await asyncio.create_subprocess_exec(executable, "--model", str(model),
                "--tokenizer", str(tokenizer), "--transport", "stdio", "--ctx-size", "128",
                "--yarn-context-extension", stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=15)
            try:
                before = await client.status()
                assert before["kv_memory_supported"] == 1
                assert before["kv_memory_dynamic"] == (allocation == "dynamic")
                assert before.get("qsa_kv_ssd_bytes", 0) == 0
                if mode == "ram":
                    assert before["kv_memory_ram_bytes"] > 0
                results = []
                for count in (90, 100, 100):
                    prompt = "ab" * count
                    async with client.generate({"messages": [{"role": "user", "content": prompt}],
                        "mfq_preformatted_prompt": prompt, "temperature": 0, "max_tokens": 4,
                        "enable_mtp": False, "stream": True}) as events:
                        chunks = [event async for event in events if event]
                    results.append("".join(choice.get("delta", {}).get("content", "")
                        for event in chunks for choice in event.get("choices", [])))
                after = await client.status()
                assert results[1] == results[2]
                if limit:
                    assert after["qsa_kv_gpu_evictions"] > 0
                    assert after["qsa_kv_ram_hits"] > 0
                    assert bool(after["qsa_kv_ssd_bytes"]) == (mode == "auto_ssd")
                return results
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    expected = asyncio.run(run("vram", 0, 0, "baseline"))
    assert asyncio.run(run(placement, gpu_limit, ram_limit, "policy")) == expected
