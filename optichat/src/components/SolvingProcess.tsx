import React, { useMemo } from 'react';
import { AlertCircle, CheckCircle2, CircleDashed } from 'lucide-react';
import type { DesktopProgressEvent } from '../types/solver';

type SolvingProcessProps = {
  events: DesktopProgressEvent[];
};

type StepMeta = {
  id: string;
  label: string;
};

const STEP_ORDER: StepMeta[] = [
  { id: 'receive', label: '接收请求' },
  { id: 'ingest', label: '识别上传实例' },
  { id: 'decision', label: '判断是否调用求解 skill' },
  { id: 'reply', label: '生成普通回复' },
  { id: 'solve', label: '调用本地求解链' },
  { id: 'seed', label: '运行 DRL 构造初始解' },
  { id: 'lookahead', label: '执行 Lookahead' },
  { id: 'local_search', label: '执行局部搜索' },
  { id: 'finalize', label: '整理最终结果' },
  { id: 'complete', label: '完成' },
  { id: 'error', label: '失败' },
];

function byTimestamp(a: DesktopProgressEvent, b: DesktopProgressEvent): number {
  return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
}

export function SolvingProcess({ events }: SolvingProcessProps) {
  const eventMap = useMemo(() => new Map(events.map((event) => [event.stepId, event])), [events]);

  const visibleSteps = useMemo(() => {
    if (events.length === 0) {
      return STEP_ORDER.slice(0, 3);
    }

    const hasIngest = eventMap.has('ingest');
    const hasReply = eventMap.has('reply');
    const hasSolve = ['solve', 'seed', 'lookahead', 'local_search', 'finalize', 'complete'].some((id) => eventMap.has(id));
    const hasLookahead = eventMap.has('lookahead');
    const hasLocalSearch = eventMap.has('local_search');
    const hasError = eventMap.has('error');

    const stepIds = ['receive'];
    if (hasIngest) {
      stepIds.push('ingest');
    }
    stepIds.push('decision');

    if (hasReply) {
      stepIds.push('reply');
    }

    if (hasSolve) {
      stepIds.push('solve');
      if (eventMap.has('seed')) {
        stepIds.push('seed');
      }
      if (hasLookahead) {
        stepIds.push('lookahead');
      }
      if (hasLocalSearch) {
        stepIds.push('local_search');
      }
      if (eventMap.has('finalize')) {
        stepIds.push('finalize');
      }
      if (eventMap.has('complete')) {
        stepIds.push('complete');
      }
    }

    if (hasError) {
      stepIds.push('error');
    }

    return STEP_ORDER.filter((step) => stepIds.includes(step.id));
  }, [eventMap, events.length]);

  const latestEvent = useMemo(() => {
    if (events.length === 0) {
      return null;
    }
    const sorted = [...events].sort(byTimestamp);
    return sorted[sorted.length - 1] ?? null;
  }, [events]);

  const runningEvent = useMemo(() => {
    const running = events.filter((event) => event.status === 'running').sort(byTimestamp);
    return running[running.length - 1] ?? null;
  }, [events]);

  const totalSteps = Math.max(visibleSteps.filter((step) => step.id !== 'error').length, 1);
  const completedCount = visibleSteps.filter((step) => eventMap.get(step.id)?.status === 'completed').length;
  const progressValue = Math.max(0.06, Math.min(1, completedCount / totalSteps));
  const currentTitle = runningEvent?.label ?? latestEvent?.label ?? '等待开始';
  const currentDetail = runningEvent?.detail ?? latestEvent?.detail ?? '正在等待后端返回真实进度事件。';

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col py-14">
      <div className="mb-10">
        <h2 className="text-2xl font-semibold text-neutral-900">处理中</h2>
        <p className="mt-3 text-sm leading-6 text-neutral-500">这里展示的是后端真实返回的阶段事件，不再使用前端定时器模拟。</p>
      </div>

      <div className="rounded-[28px] border border-neutral-200 bg-neutral-50 px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-neutral-900">{currentTitle}</div>
            <div className="mt-1 text-sm text-neutral-500">{currentDetail}</div>
          </div>
          <div className="text-xs text-neutral-400">
            {completedCount}/{totalSteps}
          </div>
        </div>

        <div className="mt-4 h-2 overflow-hidden rounded-full bg-neutral-200">
          <div className="h-full rounded-full bg-neutral-900 transition-[width] duration-300" style={{ width: `${progressValue * 100}%` }} />
        </div>
      </div>

      <div className="mt-8 space-y-5">
        {visibleSteps.map((step) => {
          const event = eventMap.get(step.id);
          const status = event?.status ?? 'pending';

          return (
            <div key={step.id} className="flex items-start gap-4">
              <div className="flex h-6 w-6 items-center justify-center">
                {status === 'completed' ? (
                  <CheckCircle2 className="text-emerald-600" size={22} />
                ) : status === 'running' ? (
                  <CircleDashed className="animate-spin text-neutral-900" size={22} />
                ) : status === 'error' ? (
                  <AlertCircle className="text-red-600" size={22} />
                ) : (
                  <div className="h-2.5 w-2.5 rounded-full bg-neutral-300" />
                )}
              </div>

              <div className="min-w-0 flex-1">
                <div
                  className={`text-base ${
                    status === 'completed'
                      ? 'text-neutral-500'
                      : status === 'running'
                        ? 'font-medium text-neutral-900'
                        : status === 'error'
                          ? 'font-medium text-red-700'
                          : 'text-neutral-400'
                  }`}
                >
                  {event?.label ?? step.label}
                </div>
                {event?.detail && <div className="mt-1 text-sm text-neutral-500">{event.detail}</div>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
