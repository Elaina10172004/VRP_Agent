import React, { useMemo, useState } from 'react';
import { MousePointer2, Route } from 'lucide-react';
import { buildSolveViewModel } from '../lib/solverView';
import type { DesktopSolveResponse } from '../types/solver';

type MapVisualizationProps = {
  solveResponse: DesktopSolveResponse;
  highlightedRouteId: string | null;
  onHighlightRoute: (routeId: string | null) => void;
};

type Bounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

function buildBounds(points: Array<{ x: number; y: number }>): Bounds {
  if (points.length === 0) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function project(value: number, minValue: number, maxValue: number, canvasSize: number, padding: number): number {
  const span = Math.max(maxValue - minValue, 1e-6);
  return padding + ((value - minValue) / span) * (canvasSize - padding * 2);
}

function formatNumber(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, '');
}

export function MapVisualization({ solveResponse, highlightedRouteId, onHighlightRoute }: MapVisualizationProps) {
  const view = useMemo(() => buildSolveViewModel(solveResponse), [solveResponse]);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const nodeLookup = useMemo(() => new Map(view.nodes.map((node) => [node.id, node])), [view.nodes]);
  const bounds = useMemo(() => buildBounds(view.nodes), [view.nodes]);

  const nodeColorLookup = useMemo(() => {
    const mapping = new Map<string, string>();
    for (const route of view.routes) {
      for (const nodeId of route.nodeIds) {
        if (nodeId !== 'depot' && !mapping.has(nodeId)) {
          mapping.set(nodeId, route.color);
        }
      }
    }
    return mapping;
  }, [view.routes]);

  const highlightedNodeIds = useMemo(() => {
    const route = view.routes.find((item) => item.id === highlightedRouteId);
    return new Set(route?.nodeIds ?? []);
  }, [highlightedRouteId, view.routes]);

  const hoveredNode = hoveredNodeId ? nodeLookup.get(hoveredNodeId) ?? null : null;
  const customerCount = view.nodes.filter((node) => !node.isDepot).length;
  const routeCount = view.routes.length;

  return (
    <div className="flex h-full w-full flex-col bg-white">
      <div className="grid gap-3 border-b border-neutral-200 bg-white px-4 py-3 md:grid-cols-4">
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3">
          <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">最终距离</div>
          <div className="mt-1 text-xl font-semibold text-neutral-900">{formatNumber(view.finalDistance)}</div>
        </div>
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3">
          <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">改进幅度</div>
          <div className="mt-1 text-xl font-semibold text-neutral-900">{formatNumber(view.improvement)}</div>
        </div>
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3">
          <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">客户数量</div>
          <div className="mt-1 text-xl font-semibold text-neutral-900">{customerCount}</div>
        </div>
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3">
          <div className="text-xs uppercase tracking-[0.16em] text-neutral-400">路线数量</div>
          <div className="mt-1 text-xl font-semibold text-neutral-900">{routeCount}</div>
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.10),_transparent_35%),radial-gradient(circle_at_bottom_right,_rgba(245,158,11,0.10),_transparent_30%),linear-gradient(180deg,#fafafa_0%,#f4f4f5_100%)]">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.06]"
          style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, black 1px, transparent 0)', backgroundSize: '34px 34px' }}
        />

        <div className="absolute left-4 top-4 z-10 flex max-w-[320px] flex-col gap-3">
          <div className="rounded-2xl border border-white/70 bg-white/88 px-4 py-3 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 text-sm font-semibold text-neutral-900">
              <Route size={16} />
              路线高亮
            </div>
            <div className="mt-1 text-xs leading-5 text-neutral-500">将鼠标移到下方路线卡片或地图线段上，只强调当前路线。</div>
          </div>

          <div className="rounded-2xl border border-white/70 bg-white/88 px-4 py-3 shadow-sm backdrop-blur">
            {hoveredNode ? (
              <div className="space-y-1.5 text-sm text-neutral-700">
                <div className="font-semibold text-neutral-900">{hoveredNode.label}</div>
                <div>
                  坐标：({formatNumber(hoveredNode.x)}, {formatNumber(hoveredNode.y)})
                </div>
                {hoveredNode.demand !== null && <div>需求：{hoveredNode.demand}</div>}
                {hoveredNode.timeWindow && <div>时间窗：[{hoveredNode.timeWindow[0]}, {hoveredNode.timeWindow[1]}]</div>}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-neutral-500">
                <MousePointer2 size={16} />
                悬停节点后显示坐标、需求和时间窗。
              </div>
            )}
          </div>
        </div>

        <svg viewBox="0 0 1000 720" className="h-full w-full" preserveAspectRatio="xMidYMin meet">
          {view.routes.map((route) =>
            route.nodeIds.slice(1).map((nodeId, index) => {
              const fromNode = nodeLookup.get(route.nodeIds[index]);
              const toNode = nodeLookup.get(nodeId);
              if (!fromNode || !toNode) {
                return null;
              }

              const x1 = project(fromNode.x, bounds.minX, bounds.maxX, 1000, 72);
              const y1 = project(fromNode.y, bounds.minY, bounds.maxY, 720, 72);
              const x2 = project(toNode.x, bounds.minX, bounds.maxX, 1000, 72);
              const y2 = project(toNode.y, bounds.minY, bounds.maxY, 720, 72);
              const isDimmed = highlightedRouteId !== null && highlightedRouteId !== route.id;
              const isHighlighted = highlightedRouteId === route.id;

              return (
                <line
                  key={`${route.id}-${index}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={route.color}
                  strokeWidth={isHighlighted ? '5.2' : '3.5'}
                  strokeLinecap="round"
                  opacity={isDimmed ? 0.12 : isHighlighted ? 0.98 : 0.76}
                  onMouseEnter={() => onHighlightRoute(route.id)}
                  onMouseLeave={() => onHighlightRoute(null)}
                />
              );
            }),
          )}

          {view.nodes.map((node) => {
            const x = project(node.x, bounds.minX, bounds.maxX, 1000, 72);
            const y = project(node.y, bounds.minY, bounds.maxY, 720, 72);
            const routeColor = node.isDepot ? '#dc2626' : nodeColorLookup.get(node.id) ?? '#404040';
            const isHovered = hoveredNodeId === node.id;
            const isDimmed = highlightedRouteId !== null && !highlightedNodeIds.has(node.id);

            return (
              <g
                key={node.id}
                transform={`translate(${x}, ${y})`}
                onMouseEnter={() => setHoveredNodeId(node.id)}
                onMouseLeave={() => setHoveredNodeId((current) => (current === node.id ? null : current))}
                className="cursor-pointer"
                opacity={isDimmed ? 0.22 : 1}
              >
                <title>
                  {node.label}
                  {`\n`}坐标: ({formatNumber(node.x)}, {formatNumber(node.y)})
                  {node.demand !== null ? `\n需求: ${node.demand}` : ''}
                  {node.timeWindow ? `\n时间窗: [${node.timeWindow[0]}, ${node.timeWindow[1]}]` : ''}
                </title>

                {node.isDepot ? (
                  <>
                    <rect x="-11" y="-11" width="22" height="22" fill={routeColor} rx="5" opacity={0.18} />
                    <rect x="-8" y="-8" width="16" height="16" fill={routeColor} rx="4" />
                  </>
                ) : (
                  <>
                    <circle r={isHovered ? '10' : '8'} fill="#ffffff" stroke={routeColor} strokeWidth={isHovered ? '3.2' : '2.4'} />
                    <circle r="3.2" fill={routeColor} />
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
