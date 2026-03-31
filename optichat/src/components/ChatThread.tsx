import React, { useMemo } from 'react';
import { CircleDashed, FileText, Sparkles } from 'lucide-react';
import { buildSolveViewModel } from '../lib/solverView';
import type { AssistantSolveTurn, ChatSession } from '../lib/history';

type ChatThreadProps = {
  session: ChatSession;
  onOpenResult: (turn: AssistantSolveTurn) => void;
  isAssistantLoading?: boolean;
};

export function ChatThread({ session, onOpenResult, isAssistantLoading = false }: ChatThreadProps) {
  const assistantViews = useMemo(() => {
    const mapping = new Map<string, ReturnType<typeof buildSolveViewModel>>();
    for (const turn of session.turns) {
      if (turn.role === 'assistant' && turn.kind === 'solve') {
        mapping.set(turn.id, buildSolveViewModel(turn.solveResponse));
      }
    }
    return mapping;
  }, [session.turns]);

  return (
    <div className="flex-1 overflow-y-auto pr-1">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-5 pb-6">
        {session.turns.map((turn) => {
          if (turn.role === 'user') {
            return (
              <div key={turn.id} className="max-w-[78%] self-end rounded-[28px] rounded-br-md bg-neutral-900 px-5 py-4 text-white shadow-sm">
                {turn.uploadedFileNames.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-2">
                    {turn.uploadedFileNames.map((fileName, index) => (
                      <div key={`${fileName}-${index}`} className="inline-flex max-w-full items-center gap-2 rounded-2xl bg-white/10 px-3 py-2 text-xs text-neutral-200">
                        <FileText size={14} />
                        <span className="truncate">{fileName}</span>
                      </div>
                    ))}
                  </div>
                )}
                {turn.text.trim() ? (
                  <div className="whitespace-pre-wrap text-sm leading-7">{turn.text}</div>
                ) : (
                  <div className="text-sm leading-7 text-neutral-300">已上传附件。</div>
                )}
              </div>
            );
          }

          if (turn.kind === 'reply') {
            return (
              <div key={turn.id} className="max-w-[82%] rounded-[28px] rounded-bl-md border border-neutral-200 bg-white px-5 py-4 text-neutral-800 shadow-sm">
                <div className="whitespace-pre-wrap text-sm leading-7">{turn.text}</div>
              </div>
            );
          }

          const view = assistantViews.get(turn.id);
          if (!view) {
            return null;
          }

          return (
            <div key={turn.id} className="max-w-[86%] rounded-[28px] rounded-bl-md border border-neutral-200 bg-white px-5 py-4 text-neutral-800 shadow-sm">
              <div className="flex items-center gap-2 text-sm font-semibold text-neutral-900">
                <Sparkles size={16} />
                {turn.solveResponse.batchItems && turn.solveResponse.batchItems.length > 1
                  ? `已完成批量求解 (${turn.solveResponse.batchItems.length} 个实例)`
                  : '已完成求解'}
              </div>

              {turn.solveResponse.batchItems && turn.solveResponse.batchItems.length > 1 ? (
                <div className="mt-3 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">实例数量</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{turn.solveResponse.batchItems.length}</div>
                  </div>
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">求解模式</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{view.modeLabel}</div>
                  </div>
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">总耗时</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{(turn.solveResponse.durationMs / 1000).toFixed(2)} 秒</div>
                  </div>
                </div>
              ) : (
                <div className="mt-3 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">问题类型</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{view.problemTypeLabel}</div>
                  </div>
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">求解模式</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{view.modeLabel}</div>
                  </div>
                  <div className="rounded-2xl bg-neutral-50 px-3 py-3">
                    <div className="text-xs text-neutral-400">最终距离</div>
                    <div className="mt-1 text-sm font-medium text-neutral-900">{view.finalDistance.toFixed(3).replace(/\.?0+$/, '')}</div>
                  </div>
                </div>
              )}

              <div className="mt-4 flex items-center justify-end gap-3">
                <button
                  onClick={() => onOpenResult(turn)}
                  className="rounded-2xl border border-neutral-200 px-3 py-2 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-50 hover:text-neutral-900"
                >
                  查看结果
                </button>
              </div>
            </div>
          );
        })}

        {isAssistantLoading && (
          <div className="max-w-[82%] rounded-[28px] rounded-bl-md border border-neutral-200 bg-white px-5 py-4 text-neutral-800 shadow-sm">
            <div className="flex items-center gap-3 text-sm text-neutral-500">
              <CircleDashed size={18} className="animate-spin" />
              <span>正在思考…</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
