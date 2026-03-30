import React, { useEffect, useState } from 'react';
import { SlidersHorizontal, X } from 'lucide-react';
import { defaultSettings, sanitizeSettings } from '../lib/settings';
import type { AppSettings } from '../types/settings';

type SettingsModalProps = {
  isOpen: boolean;
  settings: AppSettings;
  onClose: () => void;
  onSave: (settings: AppSettings) => Promise<void> | void;
};

type ToggleRowProps = {
  title: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
};

function ToggleRow({ title, description, checked, onChange }: ToggleRowProps) {
  return (
    <label className="flex items-center justify-between rounded-2xl border border-neutral-200 px-4 py-3">
      <div>
        <div className="text-sm font-medium text-neutral-700">{title}</div>
        <div className="text-xs text-neutral-500">{description}</div>
      </div>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-neutral-900"
      />
    </label>
  );
}

export function SettingsModal({ isOpen, settings, onClose, onSave }: SettingsModalProps) {
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setDraft(settings);
    }
  }, [isOpen, settings]);

  if (!isOpen) {
    return null;
  }

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave(sanitizeSettings(draft));
      onClose();
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-6">
      <div className="w-full max-w-3xl rounded-3xl border border-neutral-200 bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-neutral-200 px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-neutral-100 text-neutral-700">
              <SlidersHorizontal size={18} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-neutral-900">设置</h2>
              <p className="text-sm text-neutral-500">配置 OpenAI 连接以及本地求解链默认参数。</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-xl p-2 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-900"
            title="关闭设置"
          >
            <X size={18} />
          </button>
        </div>

        <div className="grid gap-8 px-6 py-6 lg:grid-cols-2">
          <section className="space-y-4">
            <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-neutral-500">OpenAI</h3>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">Base URL</span>
              <input
                value={draft.openaiBaseUrl}
                onChange={(event) => setDraft((previous) => ({ ...previous, openaiBaseUrl: event.target.value }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                placeholder="https://api.openai.com/v1"
              />
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">API Key</span>
              <input
                type="password"
                value={draft.openaiApiKey}
                onChange={(event) => setDraft((previous) => ({ ...previous, openaiApiKey: event.target.value }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                placeholder="sk-..."
              />
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">模型</span>
              <input
                value={draft.openaiModel}
                onChange={(event) => setDraft((previous) => ({ ...previous, openaiModel: event.target.value }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                placeholder="gpt-5.4-mini"
              />
            </label>
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-neutral-500">求解默认参数</h3>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">DRL 采样数 K</span>
              <input
                type="number"
                min={1}
                max={2048}
                value={draft.drlSamples}
                onChange={(event) => setDraft((previous) => ({ ...previous, drlSamples: Number(event.target.value) }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
              />
            </label>

            <ToggleRow
              title="启用 Lookahead"
              description="思考模式下在 DRL 初始解之后继续做 Lookahead。"
              checked={draft.enableLookahead}
              onChange={(checked) => setDraft((previous) => ({ ...previous, enableLookahead: checked }))}
            />

            <div className="grid grid-cols-2 gap-4">
              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">Lookahead 深度</span>
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={draft.lookaheadDepth}
                  onChange={(event) => setDraft((previous) => ({ ...previous, lookaheadDepth: Number(event.target.value) }))}
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>

              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">Beam 宽度</span>
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={draft.lookaheadBeamWidth}
                  onChange={(event) => setDraft((previous) => ({ ...previous, lookaheadBeamWidth: Number(event.target.value) }))}
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>
            </div>

            <ToggleRow
              title="启用局部搜索"
              description="思考模式下在 Lookahead 之后继续做局部搜索。"
              checked={draft.enableLocalSearch}
              onChange={(checked) => setDraft((previous) => ({ ...previous, enableLocalSearch: checked }))}
            />

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">局部搜索轮数</span>
              <input
                type="number"
                min={1}
                max={500}
                value={draft.localSearchRounds}
                onChange={(event) => setDraft((previous) => ({ ...previous, localSearchRounds: Number(event.target.value) }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
              />
            </label>

            <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3 text-xs leading-6 text-neutral-500">
              快速模式默认会做 8 个随机种子并行，每个种子内部使用 DRL k=128，并从结果中选最优。
              <br />
              思考模式默认使用单种子 DRL k=128，并继续执行 Lookahead；如果这里打开了局部搜索，也会继续衔接。
            </div>
          </section>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-neutral-200 px-6 py-5">
          <button
            onClick={() => setDraft(defaultSettings)}
            className="rounded-2xl border border-neutral-200 px-4 py-2.5 text-sm font-medium text-neutral-600 transition-colors hover:bg-neutral-50 hover:text-neutral-900"
          >
            恢复默认
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="rounded-2xl border border-neutral-200 px-4 py-2.5 text-sm font-medium text-neutral-600 transition-colors hover:bg-neutral-50 hover:text-neutral-900"
            >
              取消
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSaving ? '保存中...' : '保存设置'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
