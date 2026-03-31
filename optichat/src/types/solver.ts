import type { AppSettings } from './settings';

export type SolveMode = 'quick' | 'thinking';
export type ProblemType = 'tsp' | 'cvrp' | 'cvrptw';
export type PayloadSource = 'json' | 'llm' | 'upload';

export type SolverPayload = {
  problem_type: ProblemType;
  instance: Record<string, unknown>;
  config?: Record<string, unknown>;
  starting_solution?: SolverSolution;
};

export type SolverSolution = {
  problem_type: ProblemType;
  distance: number;
  tour?: number[];
  closed_tour?: number[];
  raw_sequence?: number[];
  routes?: number[][];
  checkpoint?: string;
  meta?: Record<string, unknown>;
};

export type SolverPipelineMeta = {
  mode: 'fast' | 'hybrid';
  objective?: {
    primary?: 'distance' | 'vehicle_count';
    prioritize_vehicle_count?: boolean;
    distance_weight?: number;
    duration_weight?: number;
    vehicle_fixed_cost?: number;
    vehicle_count_weight?: number;
    overtime_penalty?: number;
    lateness_penalty?: number;
    unserved_penalty?: number;
  };
  drl_samples: number;
  seed_policy: string;
  seed_trials: number;
  candidate_count_per_seed: number;
  seed_candidate_distances?: number[] | null;
  seed_candidate_scores?: number[] | null;
  enable_lookahead: boolean;
  lookahead_depth: number;
  lookahead_beam_width: number;
  lookahead_top_k?: number;
  lookahead_confident_prob?: number;
  lookahead_uncertain_chunk_size?: number;
  lookahead_candidate_distances?: number[] | null;
  enable_local_search: boolean;
  local_search_rounds: number;
  local_search_candidate_distances?: number[] | null;
  gpu_budget_mb?: number;
  fast_single_instance_vram_mb?: number;
  thinking_single_instance_vram_mb?: number;
  default_instance_parallelism?: number;
  default_instance_parallelism_by_mode?: {
    fast: number;
    hybrid: number;
  };
  tool_plan?: string[];
  tool_trace?: Array<Record<string, unknown>>;
  final_validation?: Record<string, unknown>;
  final_score?: {
    generalized_cost: number;
    vehicle_count: number;
    distance: number;
    duration: number;
    ranking_key: number[];
  };
  final_analysis?: {
    summary?: Record<string, unknown>;
    routes?: Array<Record<string, unknown>>;
    hotspots?: Record<string, unknown>;
    objective?: Record<string, unknown>;
  };
  warmup_analysis?: {
    summary?: Record<string, unknown>;
    routes?: Array<Record<string, unknown>>;
    hotspots?: Record<string, unknown>;
    objective?: Record<string, unknown>;
  };
  refinement_strategy?: Record<string, unknown>;
};

export type SolverResult = {
  problem_type: ProblemType;
  seed_solution: SolverSolution;
  lookahead_solution: SolverSolution | null;
  local_search_solution: SolverSolution | null;
  final_solution: SolverSolution;
  meta: SolverPipelineMeta;
};

export type DesktopUploadedFileRef = {
  path: string;
  name: string;
};

export type DesktopPreparedIngestResult = {
  path: string;
  ingestResult: DesktopIngestResult;
};

export type DesktopIngestRequest = {
  requestId: string;
  path: string;
};

export type ConversationTurn = {
  role: 'user' | 'assistant';
  content: string;
};

export type DesktopSolveRequest = {
  requestId: string;
  text: string;
  mode: SolveMode;
  settings: AppSettings;
  uploadedFile?: DesktopUploadedFileRef | null;
  uploadedFiles?: DesktopUploadedFileRef[] | null;
  preparsedIngestResults?: DesktopPreparedIngestResult[] | null;
  conversation?: ConversationTurn[];
  agentPreviousResponseId?: string | null;
};

export type DesktopIngestResult = {
  payload: SolverPayload;
  detected_format: string;
  source_path: string;
  saved_directory: string;
  payload_json_path: string;
  canonical_path: string;
  original_copy_path: string;
  summary: {
    problem_type: ProblemType;
    detected_format: string;
    node_count: number;
    capacity?: number;
    service_time_type?: 'scalar' | 'per_customer';
  };
};

export type RuntimeDefaults = {
  gpuBudgetMb: number;
  fastSingleInstanceVramMb: number;
  thinkingSingleInstanceVramMb: number;
  fastMaxParallelInstances: number;
  thinkingMaxParallelInstances: number;
};

export type DesktopBatchSolveItem = {
  fileName: string;
  payload: SolverPayload;
  payloadSource: PayloadSource;
  result: SolverResult;
  durationMs: number;
  structuredText?: string | null;
  ingestResult?: DesktopIngestResult | null;
};

export type DesktopSolveResponse = {
  payload: SolverPayload;
  payloadSource: PayloadSource;
  result: SolverResult;
  durationMs: number;
  structuredText?: string | null;
  ingestResult?: DesktopIngestResult | null;
  runtimeDefaults?: RuntimeDefaults;
  batchItems?: DesktopBatchSolveItem[] | null;
};

export type DesktopReplyResponse = {
  kind: 'reply';
  message: string;
  durationMs: number;
  suggestedTitle?: string | null;
  agentPreviousResponseId?: string | null;
};

export type DesktopSolveSkillResponse = {
  kind: 'solve';
  solveResponse: DesktopSolveResponse;
  durationMs: number;
  suggestedTitle?: string | null;
  agentPreviousResponseId?: string | null;
};

export type DesktopAgentResponse = DesktopReplyResponse | DesktopSolveSkillResponse;

export type DesktopOpenFileResponse = {
  path: string;
  name: string;
  content: string;
};

export type DesktopSaveJsonRequest = {
  defaultFileName: string;
  data: unknown;
};

export type DesktopProgressEvent = {
  requestId: string;
  stepId: string;
  label: string;
  status: 'running' | 'completed' | 'error';
  detail?: string | null;
  timestamp: string;
};
