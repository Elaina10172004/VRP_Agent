import type { DesktopSolveResponse, ProblemType, SolverPipelineMeta, SolverSolution } from '../types/solver';

type XY = [number, number];

export type MapNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  isDepot: boolean;
  demand: number | null;
  timeWindow: [number, number] | null;
};

export type MapRoute = {
  id: string;
  label: string;
  color: string;
  nodeIds: string[];
};

export type SolveViewModel = {
  problemTypeLabel: string;
  modeLabel: string;
  payloadSourceLabel: string;
  nodes: MapNode[];
  routes: MapRoute[];
  finalDistance: number;
  finalVehicleCount: number;
  finalGeneralizedCost: number | null;
  objectiveSummary: string;
  seedDistance: number;
  lookaheadDistance: number | null;
  localSearchDistance: number | null;
  improvement: number;
  routeLines: string[];
  stageLines: string[];
};

const ROUTE_COLORS = ['#2563eb', '#16a34a', '#dc2626', '#ca8a04', '#9333ea', '#0891b2', '#ea580c'];

function asNumber(value: unknown, fallback = 0): number {
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asNumberList(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => asNumber(item));
}

function asPointList(value: unknown): XY[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => (Array.isArray(item) && item.length >= 2 ? [asNumber(item[0]), asNumber(item[1])] : null))
    .filter((item): item is XY => item !== null);
}

function asTimeWindows(value: unknown): Array<[number, number] | null> {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => {
    if (!Array.isArray(item) || item.length < 2) {
      return null;
    }
    return [asNumber(item[0]), asNumber(item[1])];
  });
}

function formatNumber(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, '');
}

function getProblemTypeLabel(problemType: ProblemType): string {
  if (problemType === 'tsp') {
    return 'TSP 旅行商问题';
  }
  if (problemType === 'cvrp') {
    return 'CVRP 容量约束车辆路径';
  }
  return 'CVRPTW 时间窗车辆路径';
}

function getRouteLabel(problemType: ProblemType, index: number): string {
  return problemType === 'tsp' ? '回路' : `路线 ${index + 1}`;
}

function getSolutionDistance(solution: SolverSolution | null): number | null {
  return solution ? asNumber(solution.distance) : null;
}

function getPayloadSourceLabel(response: DesktopSolveResponse): string {
  if (response.payloadSource === 'upload') {
    return '上传实例';
  }
  if (response.payloadSource === 'json') {
    return '直接输入 JSON';
  }
  return '模型结构化';
}

function buildObjectiveSummary(meta: SolverPipelineMeta): string {
  const objective = meta.objective;
  if (!objective) {
    return '距离优先';
  }

  const segments = [objective.primary === 'vehicle_count' ? '车辆数优先' : '距离优先'];
  if (objective.prioritize_vehicle_count) {
    segments.push('先尽量减车');
  }
  if (objective.vehicle_count_weight) {
    segments.push(`车辆数权重 ${formatNumber(objective.vehicle_count_weight)}`);
  }
  if (objective.vehicle_fixed_cost) {
    segments.push(`车辆固定成本 ${formatNumber(objective.vehicle_fixed_cost)}`);
  }
  if (objective.distance_weight && objective.distance_weight !== 1) {
    segments.push(`距离权重 ${formatNumber(objective.distance_weight)}`);
  }
  if (objective.duration_weight) {
    segments.push(`时长权重 ${formatNumber(objective.duration_weight)}`);
  }
  if (objective.lateness_penalty) {
    segments.push(`迟到惩罚 ${formatNumber(objective.lateness_penalty)}`);
  }
  if (objective.overtime_penalty) {
    segments.push(`超时惩罚 ${formatNumber(objective.overtime_penalty)}`);
  }
  if (objective.unserved_penalty) {
    segments.push(`漏服务惩罚 ${formatNumber(objective.unserved_penalty)}`);
  }
  return segments.join(' · ');
}

function findRouteAnalysis(meta: SolverPipelineMeta, routeIndex: number): Record<string, unknown> | null {
  const routes = meta.final_analysis?.routes;
  if (!Array.isArray(routes)) {
    return null;
  }
  return (routes[routeIndex] as Record<string, unknown> | undefined) ?? null;
}

function buildTspView(response: DesktopSolveResponse): Pick<SolveViewModel, 'nodes' | 'routes' | 'routeLines'> {
  const points = asPointList(response.payload.instance.points);
  const finalSolution = response.result.final_solution;
  const closedTour =
    finalSolution.closed_tour ??
    (() => {
      const tour = finalSolution.tour ?? [];
      return tour.length > 0 ? [...tour, tour[0]] : [];
    })();

  const nodes: MapNode[] = points.map(([x, y], index) => ({
    id: `node-${index}`,
    label: `节点 ${index}`,
    x,
    y,
    isDepot: false,
    demand: null,
    timeWindow: null,
  }));

  const routes: MapRoute[] = [
    {
      id: 'route-0',
      label: '回路',
      color: ROUTE_COLORS[0],
      nodeIds: closedTour.map((node) => `node-${node}`),
    },
  ];

  const routeLines = [
    (finalSolution.tour ?? []).length > 0
      ? `回路：${(finalSolution.tour ?? []).map((node) => `节点 ${node}`).join(' -> ')}`
      : '回路：无可视化路径',
  ];

  return { nodes, routes, routeLines };
}

function buildVrpView(response: DesktopSolveResponse): Pick<SolveViewModel, 'nodes' | 'routes' | 'routeLines'> {
  const depot = Array.isArray(response.payload.instance.depot_xy) ? response.payload.instance.depot_xy : [0, 0];
  const depotPoint: XY = [asNumber(depot[0]), asNumber(depot[1])];
  const points = asPointList(response.payload.instance.node_xy);
  const demands = asNumberList(response.payload.instance.node_demand);
  const timeWindows = asTimeWindows(response.payload.instance.node_tw);
  const rawRoutes = response.result.final_solution.routes ?? [];

  const nodes: MapNode[] = [
    {
      id: 'depot',
      label: '仓库',
      x: depotPoint[0],
      y: depotPoint[1],
      isDepot: true,
      demand: null,
      timeWindow: null,
    },
    ...points.map(([x, y], index) => ({
      id: `node-${index}`,
      label: `客户 ${index + 1}`,
      x,
      y,
      isDepot: false,
      demand: demands[index] ?? null,
      timeWindow: timeWindows[index] ?? null,
    })),
  ];

  const routes: MapRoute[] = rawRoutes.map((route, index) => ({
    id: `route-${index}`,
    label: getRouteLabel(response.result.problem_type, index),
    color: ROUTE_COLORS[index % ROUTE_COLORS.length],
    nodeIds: ['depot', ...route.map((node) => `node-${node}`), 'depot'],
  }));

  const routeLines = rawRoutes.map((route, index) => {
    const body = route.map((node) => `客户 ${node + 1}`).join(' -> ');
    const analysis = findRouteAnalysis(response.result.meta, index);
    const metrics = [
      analysis?.distance !== undefined ? `距离 ${formatNumber(asNumber(analysis.distance))}` : null,
      analysis?.duration !== undefined ? `时长 ${formatNumber(asNumber(analysis.duration))}` : null,
      analysis?.load !== undefined ? `载重 ${formatNumber(asNumber(analysis.load))}` : null,
      analysis?.total_waiting_time !== undefined ? `等待 ${formatNumber(asNumber(analysis.total_waiting_time))}` : null,
    ]
      .filter(Boolean)
      .join(' · ');
    const suffix = metrics ? ` | ${metrics}` : '';
    return body
      ? `${getRouteLabel(response.result.problem_type, index)}：仓库 -> ${body} -> 仓库${suffix}`
      : `${getRouteLabel(response.result.problem_type, index)}：仓库 -> 仓库${suffix}`;
  });

  return { nodes, routes, routeLines };
}

export function buildSolveViewModel(response: DesktopSolveResponse): SolveViewModel {
  const { result } = response;
  const baseView = result.problem_type === 'tsp' ? buildTspView(response) : buildVrpView(response);
  const seedDistance = asNumber(result.seed_solution.distance);
  const lookaheadDistance = getSolutionDistance(result.lookahead_solution);
  const localSearchDistance = getSolutionDistance(result.local_search_solution);
  const finalDistance = asNumber(result.final_solution.distance);
  const finalVehicleCount = asNumber(result.meta.final_score?.vehicle_count, baseView.routes.length);
  const finalGeneralizedCost = result.meta.final_score ? asNumber(result.meta.final_score.generalized_cost) : null;
  const improvement = seedDistance - finalDistance;

  const stageLines = [
    `DRL 初始解：距离 ${formatNumber(seedDistance)}${result.meta.seed_candidate_scores?.[0] !== undefined ? ` · 分数 ${formatNumber(asNumber(result.meta.seed_candidate_scores[0]))}` : ''}`,
    ...(lookaheadDistance === null ? [] : [`Lookahead：距离 ${formatNumber(lookaheadDistance)}`]),
    ...(localSearchDistance === null ? [] : [`局部搜索：距离 ${formatNumber(localSearchDistance)}`]),
    `最终解：距离 ${formatNumber(finalDistance)} · 车辆 ${finalVehicleCount}${finalGeneralizedCost === null ? '' : ` · 分数 ${formatNumber(finalGeneralizedCost)}`}`,
  ];

  return {
    problemTypeLabel: getProblemTypeLabel(result.problem_type),
    modeLabel: result.meta.mode === 'fast' ? '快速' : '思考',
    payloadSourceLabel: getPayloadSourceLabel(response),
    nodes: baseView.nodes,
    routes: baseView.routes,
    finalDistance,
    finalVehicleCount,
    finalGeneralizedCost,
    objectiveSummary: buildObjectiveSummary(result.meta),
    seedDistance,
    lookaheadDistance,
    localSearchDistance,
    improvement,
    routeLines: baseView.routeLines,
    stageLines,
  };
}
