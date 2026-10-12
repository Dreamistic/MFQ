import { type FormEvent, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../../app/formatters';
import { Icon } from '../../app/display';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelDirectoryList } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';

export function DownloadDirectoryControl({ onDirectoryChange }: { onDirectoryChange(path: string | undefined): void }) {
  const { tr } = useSettings();
  const { connectionRevision } = useRuntime();
  const connectionScope = useConnectionScope();
  const [directory, setDirectory] = useState<string | null>(null);
  const [browser, setBrowser] = useState<ModelDirectoryList | null>(null);
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const trigger = useRef<HTMLElement | null>(null);
  const request = useRef(0);
  useEffect(() => {
    let disposed = false;
    setDirectory(null); onDirectoryChange(undefined); setOpen(false); setBrowser(null); setLoading(false); setSaving(false); setError('');
    void modelsApi.downloadDirectory().then(value => {
      if (!disposed) { setDirectory(value.actual_directory); onDirectoryChange(value.actual_directory); }
    }).catch(cause => { if (!disposed) toast.error(errorMessage(cause, tr)); });
    return () => { disposed = true; request.current += 1; };
  }, [connectionRevision, onDirectoryChange]);

  async function browse(id?: string | null, location?: string | null) {
    const current = connectionScope();
    const previousPath = browser?.current_path || directory || '/';
    const revision = ++request.current;
    setLoading(true); setError('');
    try {
      const value = await modelsApi.modelDirectories(id, location);
      if (!current() || request.current !== revision) return;
      setBrowser(value); setPath(value.current_path || location || '/');
    } catch (cause) {
      if (current() && request.current === revision) { setPath(previousPath); setError(errorMessage(cause, tr)); }
    } finally {
      if (current() && request.current === revision) setLoading(false);
    }
  }
  async function choose() {
    if (!browser?.current_id || saving) return;
    const current = connectionScope();
    setSaving(true); setError('');
    try {
      const value = await modelsApi.configureDownloadDirectory(browser.current_id);
      if (!current()) return;
      setDirectory(value.actual_directory); onDirectoryChange(value.actual_directory); setOpen(false);
      toast.success(tr('下载目录已保存', 'Download directory saved'));
    } catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
    finally { if (current()) setSaving(false); }
  }
  async function finder() {
    if (!browser?.current_id) return;
    const current = connectionScope();
    try { await modelsApi.openModelDirectoryInFinder(browser.current_id); }
    catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
  }
  async function jump(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (path.trim()) await browse(null, path.trim());
  }

  return <>
    <div className="download-directory-control">
      <span>{tr('下载目录', 'Download folder')}</span>
      <code title={directory || undefined}>{directory || '--'}</code>
      <button type="button" disabled={!directory || saving} onClick={event => {
        trigger.current = event.currentTarget; setOpen(true); setPath(directory!); void browse(null, directory);
      }}><Icon name="folder" size={14} />{tr('更改', 'Change')}</button>
    </div>
    <ModelDirectoryDialog title={tr('选择下载文件夹', 'Choose download folder')} catalog={{
      modelBrowser: browser, modelBrowserOpen: open, modelBrowserError: error, modelBrowserLoading: loading,
      modelFilesMode: false, setModelBrowserOpen: setOpen, modelBrowserTriggerRef: trigger, busy: saving || loading,
      jumpToModelDirectory: jump, openModelDirectory: browse, modelDirectoryPath: path, setModelDirectoryPath: setPath,
      registerCurrentModelDirectory: choose, openCurrentDirectoryInFinder: finder,
    }} />
  </>;
}
