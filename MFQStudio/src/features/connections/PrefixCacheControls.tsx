import { useEffect, useRef, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { ApiError } from '../../shared/api/client';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { SettingRow } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';

type CacheSettings = { prefix_directory: string | null; prefix_disk_limit_bytes: number | null };

export function PrefixCacheControls({ budget }: { budget: number }) {
  const { addJob, connectionRevision } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr } = useSettings();
  const [applied, setApplied] = useState<CacheSettings | null>(null);
  const [manualDirectory, setManualDirectory] = useState(false);
  const [directory, setDirectory] = useState('');
  const [actual, setActual] = useState('');
  const [manualBudget, setManualBudget] = useState(false);
  const [draft, setDraft] = useState('100');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [operationId, setOperationId] = useState('');
  const previous = useRef<CacheSettings | null>(null);
  const jobs = useJobStore(state => state.jobs);
  const pending = jobs.some(job => job.kind === 'runtime.memory.configure' && ['queued', 'running', 'cancelling'].includes(job.status));
  const operation = jobs.find(job => job.id === operationId);
  function restore(value: CacheSettings) {
    setApplied(value); setManualDirectory(value.prefix_directory != null); setDirectory(value.prefix_directory || actual);
    setManualBudget(value.prefix_disk_limit_bytes != null); setDraft(String((value.prefix_disk_limit_bytes ?? budget) / 2 ** 30));
  }
  useEffect(() => {
    setApplied(null); setSaving(false); setError(''); setOperationId(''); previous.current = null;
  }, [connectionRevision]);
  useEffect(() => {
    if (operation && ['failed', 'cancelled', 'interrupted'].includes(operation.status)) {
      if (previous.current) restore(previous.current);
      setError(operation.error ? errorMessage(new ApiError(400, { error: operation.error }), tr)
        : tr('前缀缓存设置未能应用，已恢复原设置。', 'Prefix cache settings could not be applied. The previous settings were restored.'));
    }
  }, [operation?.id, operation?.status]);
  useEffect(() => {
    if (pending) return;
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      setActual(policy.actual_prefix_directory || '');
      setApplied({ prefix_directory: policy.prefix_directory ?? null, prefix_disk_limit_bytes: policy.prefix_disk_limit_bytes ?? null });
      setManualDirectory(policy.prefix_directory != null); setDirectory(policy.prefix_directory || policy.actual_prefix_directory || '');
      setManualBudget(policy.prefix_disk_limit_bytes != null); setDraft(String((policy.prefix_disk_limit_bytes ?? budget) / 2 ** 30));
    }).catch(cause => { if (!disposed) setError(errorMessage(cause, tr)); });
    return () => { disposed = true; };
  }, [pending, connectionRevision, operation?.status]);
  const bytes = manualBudget ? Math.round(Number(draft) * 2 ** 30) : null;
  const valid = (!manualDirectory || directory.trim() !== '') && (!manualBudget || draft.trim() !== '' && Number.isSafeInteger(bytes) && bytes! >= 0);
  const next: CacheSettings = { prefix_directory: manualDirectory ? directory.trim() : null, prefix_disk_limit_bytes: bytes };
  const changed = applied != null && (next.prefix_directory !== applied.prefix_directory || next.prefix_disk_limit_bytes !== applied.prefix_disk_limit_bytes);
  const busy = saving || pending;
  async function apply() {
    if (!applied || !valid || !changed || busy) return;
    const current = connectionScope();
    const before = applied;
    const changes: Partial<CacheSettings> = {};
    if (next.prefix_directory !== before.prefix_directory) changes.prefix_directory = next.prefix_directory;
    if (next.prefix_disk_limit_bytes !== before.prefix_disk_limit_bytes) changes.prefix_disk_limit_bytes = next.prefix_disk_limit_bytes;
    setSaving(true); setError('');
    let submitted = false;
    try {
      const accepted = await runtimeApi.configureMemoryPolicy(changes);
      if (!current()) return;
      submitted = true; previous.current = before; setApplied(next); setOperationId(accepted.operation_id);
      const job = await jobsApi.getJob(accepted.operation_id);
      if (!current()) return;
      addJob(job);
      toast.success(tr('前缀缓存设置已提交', 'Prefix cache settings submitted'));
    } catch (cause) { if (current()) {
      if (!submitted) restore(before);
      setError(submitted ? tr('设置已提交，但暂时无法读取进度，请刷新后查看。', 'Settings were submitted, but progress is unavailable. Refresh to check.') : errorMessage(cause, tr));
    } }
    finally { if (current()) setSaving(false); }
  }
  return <>
    <SettingRow title={tr('目录', 'Directory')} detail={actual || '--'}
      trailing={<div className="server-row-actions prefix-directory-controls">
        {manualDirectory && <input aria-label={tr('前缀缓存目录', 'Prefix cache directory')} disabled={busy} value={directory}
          onChange={event => { setDirectory(event.target.value); setError(''); }} placeholder={tr('绝对目录路径', 'Absolute directory path')}
          onBlur={() => { if (!directory.trim()) { setDirectory(applied?.prefix_directory || actual); setError(tr('缓存目录不能为空，已恢复原目录。', 'The cache folder cannot be empty. The previous folder was restored.')); } }} />}
        <select aria-label={tr('前缀缓存目录模式', 'Prefix cache directory mode')} disabled={!applied || busy}
          value={manualDirectory ? 'manual' : 'managed'} onChange={event => { setManualDirectory(event.target.value === 'manual'); setError(''); }}>
          <option value="managed">{tr('由 MFQ 管理', 'Managed by MFQ')}</option><option value="manual">{tr('手动位置', 'Manual location')}</option>
        </select>
      </div>} />
    <SettingRow title={tr('SSD 总配额', 'Total SSD budget')}
      detail={tr('所有模型共用；自动使用剩余磁盘空间与现有缓存总量的一半，超额按 LRU 回收。',
        'Shared across models. Automatic uses half of free space plus existing cache; LRU reclaims overflow.')}
      trailing={<div className="server-row-actions">
        {manualBudget ? <div className="server-input-unit"><input aria-label={tr('SSD 预算上限', 'SSD budget limit')}
          className="server-number-input" type="number" min="0" step="any" disabled={busy} value={draft}
          onChange={event => { setDraft(event.target.value); setError(''); }}
          onBlur={() => { if (draft.trim() === '' || !Number.isSafeInteger(bytes) || bytes! < 0) { setDraft(String((applied?.prefix_disk_limit_bytes ?? budget) / 2 ** 30)); setError(tr('SSD 配额必须为非负数，已恢复原值。', 'The SSD budget must be nonnegative. The previous value was restored.')); } }} /><span>GiB</span></div>
          : <span className="server-managed-value">{budget > 0 ? `${formatNumber(budget / 2 ** 30, 0)} GiB` : '--'}</span>}
        <select aria-label={tr('SSD 配额模式', 'SSD budget mode')} disabled={!applied || busy}
          value={manualBudget ? 'manual' : 'automatic'} onChange={event => { setManualBudget(event.target.value === 'manual'); setError(''); }}>
          <option value="automatic">{tr('自动', 'Automatic')}</option><option value="manual">{tr('手动', 'Manual')}</option>
        </select>
      </div>} />
    <div className="memory-budget-actions prefix-cache-actions"><small role="status">{changed ? tr('未应用', 'Not applied') : ''}</small>
      <button type="button" disabled={!applied || busy || !valid || !changed} onClick={() => void apply()}>
        {busy ? tr('应用中…', 'Applying…') : tr('应用', 'Apply')}
      </button>
    </div>
    {(error || !valid) && <p className="server-apply-state" role="alert">{error || (manualDirectory && !directory.trim()
      ? tr('缓存目录不能为空。', 'The cache folder cannot be empty.') : tr('SSD 配额必须为非负数。', 'The SSD budget must be nonnegative.'))}</p>}
  </>;
}
