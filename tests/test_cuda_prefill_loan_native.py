"""Prefill may overwrite expert tails, then decode must use restored weights."""
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


def test_prefill_loan_restores_experts_and_retains_graphs(tmp_path):
    executable = os.environ.get('MFQ_CUDA_RESOURCE_TEST_RUNTIME')
    if not executable:
        pytest.skip('CUDA runtime required')
    config, weights = mtp_fixture('qwen', 1)
    config['text_config']['num_experts'] = 32
    config['text_config']['max_position_embeddings'] = 2048
    for name, value in list(weights.items()):
        if '.experts.' in name or (name.endswith(('.router.weight', '.router.bias')) and value.shape[0] == 3):
            value = value[np.arange(32) % 3].copy()
            if '.experts.' in name:
                value *= (1 + np.arange(32, dtype=np.float32) * .01).reshape((-1, 1, 1))
            weights[name] = value
        if name.endswith('.experts.gate_up.weight'):
            for role, matrix in zip(('gate', 'up'), np.split(value, 2, axis=1)):
                weights[name.replace('gate_up', role)] = matrix.copy()
            del weights[name]
    for name, value in list(weights.items()):
        if '.experts.' in name:
            experts, rows, width = value.shape
            packed = quantize(value.reshape(experts * rows, width), NintSpec(4, 8, 6))
            weights[name] = MfeTensor(value.shape, (MfePool(np.arange(experts, dtype=np.int32), packed),))
        elif '.ngram.shard.' in name:
            weights[name] = quantize(value, NintSpec(4, 8, 6))
    model = tmp_path / 'loan.mfq'
    write_qwen_fixture(model, config, weights, True)
    tokenizer = _worker_tokenizer(tmp_path)

    async def run(borrow):
        log_path = tmp_path / f'worker-{borrow}.log'
        env = dict(os.environ, MFQ_PREFILL_BORROW=borrow, MFQ_QWEN4_REUSE_GRAPHS='1',
            MFQ_KV_PLACEMENT='vram', MFQ_KV_ALLOCATION='preallocated', MFQ_KV_TURBOQUANT_BITS='0',
            MFQ_MOE_PIPELINE='1', MFQ_MOE_ROUTER_LOOKAHEAD='0', MFQ_MOE_TWO_STAGE_FFN='1',
            MFQ_MOE_PRELOAD_ALL='1', MFQ_MOE_PREFILL_LAYER='1', MFQ_MOE_PREFILL_GRAPH='0',
            MFQ_MOE_HOST_CACHE_BYTES='10485760', MFQ_RUNTIME_DISABLE_PREFIX_CACHE='1')
        with log_path.open('wb') as log:
            process = await asyncio.create_subprocess_exec(executable, '--model', str(model),
                '--tokenizer', str(tokenizer), '--transport', 'stdio', '--ctx-size', '2048',
                '--prefill-chunk-size', '1536', '--moe-gpu-cache-gb', '0.0009',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, env=env)
            client = StdioRuntimeClient(process, control_timeout_seconds=30)
            try:
                await client.status()
                captures = log_path.read_text(errors='replace').count('runtime_qwen4_graph_capture')
                outputs = []
                for prompt in ('ab'*1200, 'ba'*1100, 'abab'):
                    async with client.generate({'messages': [{'role':'user', 'content':prompt}],
                        'mfq_preformatted_prompt':prompt, 'temperature':0, 'max_tokens':12,
                        'enable_mtp':True, 'mtp_max_draft_tokens':3, 'stream':True}) as events:
                        chunks = [event async for event in events if event]
                    outputs.append(''.join(choice.get('delta', {}).get('content', '')
                        for event in chunks for choice in event.get('choices', [])))
                await client.status()
                trace = log_path.read_text(errors='replace')
                assert trace.count('runtime_qwen4_graph_capture') == captures
                assert 'cleanup_error' not in trace and 'grouped_fallback' not in trace
                if borrow == 'force':
                    assert trace.count('runtime_prefill_loan action=borrow') == 4
                    assert trace.count('runtime_prefill_loan action=return') == 4
                    assert trace.count('runtime_prefill_ring split=1') == 2
                else:
                    assert 'runtime_prefill_loan action=borrow' not in trace
                return outputs
            finally:
                with suppress(Exception):
                    await client.aclose()
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)
    assert asyncio.run(run('force')) == asyncio.run(run('0'))
