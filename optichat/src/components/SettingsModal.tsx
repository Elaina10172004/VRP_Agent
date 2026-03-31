import React, { useEffect, useState } from 'react';
import { KeyRound, SlidersHorizontal, X } from 'lucide-react';
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

  const hasStoredKey = Boolean(settings.hasStoredOpenaiApiKey);
  const keyTail = settings.openaiApiKeyLast4 ? `****${settings.openaiApiKeyLast4}` : '';

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
              <p className="text-sm text-neutral-500">配置 OpenAI 连接和默认求解参数。</p>
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
                placeholder={hasStoredKey ? '已保存，如需更换请重新输入' : 'sk-...'}
              />
            </label>

            {hasStoredKey && !draft.openaiApiKey && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs leading-6 text-emerald-700">
                <div className="flex items-center gap-2 font-medium">
                  <KeyRound size={14} />
                  已保存 API key {keyTail || ''}
                </div>
                <div className="mt-1">留空保存不会清空当前 key，只有输入新的 key 才会覆盖。</div>
              </div>
            )}

            <label className="block space-y-2">
              <span className="text-sm font-medium text-neutral-700">模型</span>
              <input
                value={draft.openaiModel}
                onChange={(event) => setDraft((previous) => ({ ...previous, openaiModel: event.target.value }))}
                className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                placeholder="gpt-5.4"
              />
            </label>
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-neutral-500">默认求解参数</h3>

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
              description="思考模式下启用 PolyNet 式解码期 lookahead。"
              checked={draft.enableLookahead}
              onChange={(checked) => setDraft((previous) => ({ ...previous, enableLookahead: checked }))}
            />

            <div className="grid grid-cols-2 gap-4">
              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">Lookahead Top-K</span>
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={draft.lookaheadTopK}
                  onChange={(event) => setDraft((previous) => ({ ...previous, lookaheadTopK: Number(event.target.value) }))}
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>

              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">置信概率阈值</span>
                <input
                  type="number"
                  min={0.5}
                  max={0.999}
                  step={0.01}
                  value={draft.lookaheadConfidentProb}
                  onChange={(event) =>
                    setDraft((previous) => ({ ...previous, lookaheadConfidentProb: Number(event.target.value) }))
                  }
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>
            </div>

            <ToggleRow
              title="思考模式启用改解"
              description="控制 thinking 模式下是否继续做多轮局部搜索和破坏修复。"
              checked={draft.enableLocalSearch}
              onChange={(checked) => setDraft((previous) => ({ ...previous, enableLocalSearch: checked }))}
            />

            <div className="grid grid-cols-2 gap-4">
              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">Fast 改解轮数</span>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={draft.fastLocalSearchRounds}
                  onChange={(event) =>
                    setDraft((previous) => ({ ...previous, fastLocalSearchRounds: Number(event.target.value) }))
                  }
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>

              <label className="block space-y-2">
                <span className="text-sm font-medium text-neutral-700">Thinking 改解轮数</span>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={draft.thinkingLocalSearchRounds}
                  onChange={(event) =>
                    setDraft((previous) => ({ ...previous, thinkingLocalSearchRounds: Number(event.target.value) }))
                  }
                  className="w-full rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-900 outline-none transition-colors focus:border-neutral-400"
                />
              </label>
            </div>

            <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3 text-xs leading-6 text-neutral-500">
              Fast 模式默认是多 seed 的 DRL 构造，再做轻量局部搜索。
              <br />
              Thinking 模式默认先做 DRL k=128，再做 decode-time lookahead，然后让模型选择算子进行多轮改进。
              <br />
              Lookahead 的 Chunk Size 已固定为 256，不再暴露给用户修改。
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
