import React, { useMemo, useState } from 'react';
import { CheckCircle2, Download, MessageSquarePlus, Plus } from 'lucide-react';
import { MapVisualization } from './MapVisualization';
import { buildSolveViewModel } from '../lib/solverView';
import type { DesktopSolveResponse } from '../types/solver';

type ResultViewProps = {
  solveResponse: DesktopSolveResponse;
  onBackToChat: () => void;
  onOpenNewChat: () => void;
};

function formatNumber(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, '');
}

function buildExportFileName(response: DesktopSolveResponse): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  if (response.batchItems && response.batchItems.length > 1) {
    return `batch-${response.batchItems.length}-${timestamp}.json`;
  }
  return `${response.result.problem_type}-${timestamp}.json`;
}

export function ResultView({ solveResponse, onBackToChat, onOpenNewChat }: ResultViewProps) {
  const [selectedBatchIndex, setSelectedBatchIndex] = useState(0);
  const [highlightedRouteId, setHighlightedRouteId] = useState<string | null>(null);

  const batchItems = solveResponse.batchItems ?? null;
  const isBatch = Boolean(batchItems && batchItems.length > 1);
  const activeItem = isBatch ? batchItems?.[Math.min(selectedBatchIndex, (batchItems?.length ?? 1) - 1)] ?? null : null;

  const activeSolveResponse = useMemo<DesktopSolveResponse>(() => {
    if (!isBatch || !activeItem) {
      return solveResponse;
    }

    return {
      ...solveResponse,
      payload: activeItem.payload,
      payloadSource: activeItem.payloadSource,
      result: activeItem.result,
      durationMs: activeItem.durationMs,
      structuredText: activeItem.structuredText ?? solveResponse.structuredText,
      ingestResult: activeItem.ingestResult ?? null,
      batchItems,
    };
  }, [activeItem, batchItems, isBatch, solveResponse]);

  const view = useMemo(() => buildSolveViewModel(activeSolveResponse), [activeSolveResponse]);
  const payloadText = useMemo(() => JSON.stringify(activeSolveResponse.payload, null, 2), [activeSolveResponse.payload]);
  const ingestSummary = activeSolveResponse.ingestResult?.summary ?? null;

  const handleExport = async () => {
    const payload = {
      exportedAt: new Date().toISOString(),
      solveResponse,
    };

    if (window.desktopApp?.files?.saveJson) {
      await window.desktopApp.files.saveJson({
        defaultFileName: buildExportFileName(solveResponse),
        data: payload,
      });
      return;
    }

    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = buildExportFileName(solveResponse);
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-5">
      {isBatch && batchItems && (
        <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-neutral-900">批量实例列表</div>
              <div className="mt-1 text-xs text-neutral-500">批量任务已并行提交，点击任一实例即可切换当前结果与地图。</div>
            </div>
            <div className="text-xs text-neutral-400">{batchItems.length} 个实例</div>
          </div>

          <div className="mt-4 flex gap-3 overflow-x-auto pb-1">
            {batchItems.map((item, index) => {
              const isActive = index === selectedBatchIndex;
              return (
                <button
                  key={`${item.fileName}-${index}`}
                  onClick={() => {
                    setSelectedBatchIndex(index);
                    setHighlightedRouteId(null);
                  }}
                  className={`min-w-[240px] max-w-[300px] flex-none rounded-2xl border px-4 py-3 text-left text-sm transition-colors ${
                    isActive
                      ? 'border-neutral-900 bg-neutral-900 text-white'
                      : 'border-neutral-200 bg-neutral-50 text-neutral-700 hover:bg-white'
                  }`}
                >
                  <div className="truncate font-medium">{item.fileName}</div>
                  <div className={`mt-1 text-xs leading-5 ${isActive ? 'text-neutral-200' : 'text-neutral-500'}`}>
                    {item.result.problem_type.toUpperCase()} · 距离 {formatNumber(item.result.final_solution.distance)}
                  </div>
                  <div className={`mt-1 text-xs ${isActive ? 'text-neutral-300' : 'text-neutral-400'}`}>
                    用时 {(item.durationMs / 1000).toFixed(2)} 秒
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="grid min-h-0 flex-1 gap-5 xl:grid-cols-[340px_minmax(0,1fr)]">
        <div className="min-h-0 overflow-y-auto pr-1">
          <div className="space-y-4">
            <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-lg font-semibold text-neutral-900">求解结果</div>
                  <div className="mt-1 text-sm text-neutral-500">
                    {isBatch
                      ? `本次共完成 ${batchItems?.length ?? 0} 个实例的批量求解。`
                      : '可以继续查看地图、导出 JSON，或返回主对话继续提问。'}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={handleExport}
                    className="inline-flex items-center gap-2 rounded-2xl border border-neutral-200 px-3 py-2 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-50 hover:text-neutral-900"
                  >
                    <Download size={16} />
                    导出 JSON
                  </button>

                  <button
                    onClick={onBackToChat}
                    className="inline-flex items-center gap-2 rounded-2xl border border-neutral-200 px-3 py-2 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-50 hover:text-neutral-900"
                  >
                    <MessageSquarePlus size={16} />
                    返回对话
                  </button>

                  <button
                    onClick={onOpenNewChat}
                    className="inline-flex items-center gap-2 rounded-2xl bg-neutral-900 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-neutral-800"
                  >
                    <Plus size={16} />
                    新对话
                  </button>
                </div>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">{isBatch ? '批量规模' : '问题类型'}</div>
                  <div className="mt-1 text-base font-semibold text-neutral-900">
                    {isBatch ? `${batchItems?.length ?? 0} 个实例` : view.problemTypeLabel}
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">求解模式</div>
                  <div className="mt-1 text-base font-semibold text-neutral-900">{view.modeLabel}</div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">{isBatch ? '当前实例' : '输入来源'}</div>
                  <div className="mt-1 text-base font-semibold text-neutral-900">{isBatch ? activeItem?.fileName ?? '-' : view.payloadSourceLabel}</div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">耗时</div>
                  <div className="mt-1 text-base font-semibold text-neutral-900">
                    {((isBatch ? solveResponse.durationMs : activeSolveResponse.durationMs) / 1000).toFixed(2)} 秒
                  </div>
                </div>
                {ingestSummary && (
                  <div>
                    <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">识别格式</div>
                    <div className="mt-1 text-base font-semibold text-neutral-900">{ingestSummary.detected_format}</div>
                  </div>
                )}
                {ingestSummary && (
                  <div>
                    <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">节点数量</div>
                    <div className="mt-1 text-base font-semibold text-neutral-900">{ingestSummary.node_count}</div>
                  </div>
                )}
                {ingestSummary?.capacity !== undefined && (
                  <div>
                    <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">容量</div>
                    <div className="mt-1 text-base font-semibold text-neutral-900">{ingestSummary.capacity}</div>
                  </div>
                )}
              </div>

              <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  {isBatch ? `当前实例最终距离：${formatNumber(view.finalDistance)}` : `最终距离：${formatNumber(view.finalDistance)}`}
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
              <div className="text-sm font-semibold text-neutral-900">求解过程</div>
              <div className="mt-3 space-y-2.5">
                {view.stageLines.map((line) => (
                  <div key={line} className="rounded-2xl bg-neutral-50 px-4 py-3 text-sm text-neutral-700">
                    {line}
                  </div>
                ))}
              </div>
              <div className="mt-3 text-sm text-neutral-500">
                相比 DRL 初始解，总改进为 <span className="font-medium text-neutral-900">{formatNumber(view.improvement)}</span>。
              </div>
            </div>

            <details className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
              <summary className="cursor-pointer text-sm font-semibold text-neutral-900">查看原始 payload</summary>
              <pre className="mt-4 overflow-x-auto rounded-2xl bg-neutral-950 p-4 text-xs leading-6 text-neutral-100">{payloadText}</pre>
            </details>
          </div>
        </div>

        <div className="min-h-[760px] overflow-hidden rounded-[30px] border border-neutral-200 bg-white shadow-sm">
          <MapVisualization
            solveResponse={activeSolveResponse}
            highlightedRouteId={highlightedRouteId}
            onHighlightRoute={setHighlightedRouteId}
          />
        </div>
      </div>

      <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
        <div className="text-sm font-semibold text-neutral-900">路线明细</div>
        <div className="mt-1 text-xs text-neutral-500">每行显示三条路线，悬停时会同步高亮地图中的对应路线。</div>

        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {view.routes.map((route, index) => {
            const isActive = highlightedRouteId === route.id;
            const isDimmed = highlightedRouteId !== null && highlightedRouteId !== route.id;

            return (
              <button
                key={route.id}
                onMouseEnter={() => setHighlightedRouteId(route.id)}
                onMouseLeave={() => setHighlightedRouteId(null)}
                className={`w-full rounded-2xl border px-4 py-3 text-left text-sm leading-6 transition-colors ${
                  isActive
                    ? 'border-neutral-900 bg-neutral-900 text-white'
                    : isDimmed
                      ? 'border-neutral-200 bg-white text-neutral-400'
                      : 'border-neutral-200 bg-neutral-50 text-neutral-700'
                }`}
              >
                <div className="mb-2 flex items-center gap-2">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: route.color }} />
                  <span className="font-medium">{route.label}</span>
                </div>
                <div className="text-xs leading-6">{view.routeLines[index]}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
