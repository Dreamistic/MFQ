import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus, RuntimeModel } from '../../shared/api/contracts/runtime';
import { OverviewPage } from './OverviewPage';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, models: [] as RuntimeModel[], instances: [],
  connectionRevision: 0, refreshRuntime: vi.fn(), listener: vi.fn().mockResolvedValue({ anthropic_port: null }) }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { runtimeListener: state.listener } }));
vi.mock('./RuntimeHero', () => ({ RuntimeHero: () => null }));
vi.mock('./MemoryHierarchy', () => ({ MemoryHierarchy: () => null }));

beforeEach(() => { state.runtime = null; state.models = []; });

async function prefixTile() {
  await act(async () => { render(<OverviewPage />); });
  return screen.getByText('Prefix reuse').closest('.metric-tile')!;
}

it.each<RuntimeStatus | null>([
  null,
  { runtime_state: 'idle', prefix_cache_supported: 0 },
  { runtime_state: 'idle', prefix_cache_supported: 1, prefix_cache_queries: 4, prefix_cache_hits: 3 },
])('shows placeholders without a loaded model (%j)', async runtime => {
  state.runtime = runtime;
  const tile = await prefixTile();
  expect(tile.querySelector('strong')).toHaveTextContent('--');
  expect(tile.querySelector('small')).toHaveTextContent('--');
  expect(tile).not.toHaveTextContent('Unavailable');
  expect(tile).not.toHaveTextContent('does not support');
});

it('retains the capability explanation for a loaded unsupported model', async () => {
  state.runtime = { model: 'model-a', prefix_cache_supported: 0 };
  const tile = await prefixTile();
  expect(tile.querySelector('strong')).toHaveTextContent('Unavailable');
  expect(tile.querySelector('small')).toHaveTextContent('The current model does not support Session KV cache');
});

it('retains cache statistics for a loaded supported model', async () => {
  state.runtime = { model: 'model-a', prefix_cache_supported: 1, prefix_cache_queries: 4, prefix_cache_hits: 3, prefix_cache_hit_tokens: 128 };
  const tile = await prefixTile();
  expect(tile.querySelector('strong')).toHaveTextContent('75%');
  expect(tile.querySelector('small')).toHaveTextContent('128 tokens restored');
});

it('recognizes an advertised loaded model when the runtime snapshot has no model name', async () => {
  state.models = [{ id: 'model-a' }];
  state.runtime = { prefix_cache_supported: 0 };
  expect((await prefixTile()).querySelector('strong')).toHaveTextContent('Unavailable');
});
