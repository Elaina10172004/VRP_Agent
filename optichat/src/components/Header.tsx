import React from 'react';
import { Settings } from 'lucide-react';

type HeaderProps = {
  state: string;
  onOpenSettings: () => void;
};

const stateLabelMap: Record<string, string> = {
  idle: '对话',
  solving: '处理中',
  result: '结果',
};

export function Header({ state, onOpenSettings }: HeaderProps) {
  return (
    <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-neutral-200 bg-white/95 px-6 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="text-lg font-semibold tracking-wide text-neutral-900">OptiChat</div>
        <div className="rounded-full border border-neutral-200 bg-neutral-50 px-2.5 py-0.5 text-xs font-medium text-neutral-500">
          {stateLabelMap[state] ?? state}
        </div>
      </div>

      <button
        onClick={onOpenSettings}
        className="rounded-xl p-2 text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-900"
        title="设置"
      >
        <Settings size={18} />
      </button>
    </header>
  );
}
