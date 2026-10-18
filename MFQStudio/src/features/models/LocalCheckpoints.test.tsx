import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelArtifact } from '../../shared/api/types';
import type { useModelCatalog } from './useModelCatalog';
import { LocalCheckpoints } from './LocalCheckpoints';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./ModelLoadProgress', () => ({ ModelLoadProgress: () => null }));

it('uses one combined MoE pressure bar without capacity details, matching the download page', () => {
  const GiB = 2 ** 30;
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'mfq',
    shard_count: 1, total_bytes: 133 * GiB, estimated_resident_weight_bytes: 83 * GiB, ssd_ple_bytes: 50 * GiB,
    estimated_weight_bytes_by_role: { dense: 2 * GiB, experts: 80 * GiB, embedding: GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-11' };
  const catalog = { runtime: { memory_architecture: 'discrete', device_memory_total_bytes: 24 * GiB,
    host_memory_total_bytes: 64 * GiB, runtime_memory_headroom_bytes: 8 * GiB },
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('progressbar', { name: 'VRAM + RAM pressure: 94.3%' })).toBeInTheDocument();
  expect(screen.getAllByRole('progressbar')).toHaveLength(1);
  expect(document.querySelector('.variant-memory-caption')).toBeNull();
  expect(screen.getByText(/SSD PLE 50/)).toBeInTheDocument();
});

it('shows the same zero shared budget as the model-loading estimate, not raw free memory', () => {
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'mfq',
    shard_count: 1, total_bytes: 2 ** 30, estimated_resident_weight_bytes: 2 ** 30,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-18' };
  const catalog = { runtime: { memory_architecture: 'unified', host_memory_available_bytes: 3 * 2 ** 30,
    runtime_memory_headroom_bytes: 0 }, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByText('Remaining available memory 0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'Estimated share of remaining available memory: No available memory' })).toBeInTheDocument();
});
