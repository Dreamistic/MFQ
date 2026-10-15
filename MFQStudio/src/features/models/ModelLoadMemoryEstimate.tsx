import { useEffect, useState } from 'react';
import type { ModelArtifact, ModelCacheProfile, RuntimeStatus } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { availableLoadMemory, estimateLoadMemory } from './loadMemoryEstimate';
import type { LoadContextDefaults } from './loadContextDefaults';

function gib(bytes: number | null) {
  return bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
}

export function ModelLoadMemoryEstimate({ models, runtime, deviceIds, value, valid, disabled }: {
  models: ModelArtifact[]; runtime?: RuntimeStatus | null; deviceIds: string[];
  value: LoadContextDefaults; valid: boolean; disabled: boolean;
}) {
  const { tr } = useSettings();
  const [selectedId, setSelectedId] = useState('');
  const [data, setData] = useState<{ key: string; profile: ModelCacheProfile | null } | null>(null);
  const choices = models.filter(model => model.complete && model.loadable);
  const selected = choices.find(model => model.id === selectedId);
  const key = selected ? `${selected.id}:${selected.modified_at}:${selected.total_bytes}` : '';
  useEffect(() => {
    if (!selected) { setSelectedId(''); return; }
    const controller = new AbortController();
    let active = true;
    void modelsApi.modelCacheProfile(selected.id, controller.signal, true)
      .then(profile => { if (active) setData({ key, profile }); })
      .catch(() => { if (active) setData({ key, profile: null }); });
    return () => { active = false; controller.abort(); };
  }, [selected?.id, key]);
  const available = availableLoadMemory(runtime, deviceIds);
  const loading = !!selected && data?.key !== key;
  const estimate = selected && data?.key === key && data.profile && valid
    ? estimateLoadMemory(selected, data.profile, value) : null;
  const exceeded = estimate != null && available != null && estimate.total > available;
  return <div className="model-load-memory-estimate">
    <p>{tr('不确定设备能否载入模型时，可在这里选择模型，结合当前可用空间查看预计占用。模型载入时可能释放部分系统占用，可用空间通常会略高于当前测量值，但增加有限。',
      'If you are unsure whether your device can load a model, select it here to compare estimated usage with currently available memory. Loading may release some system memory, so available space is often slightly higher than measured, but the increase is limited.')}</p>
    <div className="model-load-estimate-row">
      <strong>{tr('预计总占用（显存 + 内存）', 'Estimated total (VRAM + RAM)')}</strong>
      <div className="model-load-estimate-values">
        <span className="model-load-memory-available">{tr('当前剩余', 'Available now')} <strong>{gib(available)}</strong></span>
        <output aria-label={tr('预计总占用', 'Estimated total memory')} aria-live="polite"
          className={exceeded ? 'model-load-memory-exceeded' : undefined}
          title={estimate ? tr(`权重与预留 ${gib(estimate.weights)}，上下文缓存与递推状态 ${gib(estimate.caches)}（${estimate.context.toLocaleString()} tokens）。实际运行占用会随工作区变化。`,
            `Weights and reserve ${gib(estimate.weights)}, context caches and recurrent state ${gib(estimate.caches)} (${estimate.context.toLocaleString()} tokens). Runtime usage varies with workspaces.`) : undefined}>
          {!selected ? '—' : loading ? tr('估算中…', 'Estimating…') : !valid ? '—'
            : estimate ? gib(estimate.total) : tr('暂无法估算', 'Estimate unavailable')}
        </output>
        <select aria-label={tr('估算模型', 'Model to estimate')} value={selected?.id ?? ''} disabled={disabled}
          onChange={event => setSelectedId(event.target.value)}>
          <option value="">{tr('无', 'None')}</option>
          {choices.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>
      </div>
    </div>
  </div>;
}
