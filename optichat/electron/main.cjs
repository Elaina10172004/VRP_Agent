const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { app, BrowserWindow, dialog, ipcMain, safeStorage, shell } = require('electron');

const isDev = process.env.OPTICHAT_DEV === 'true';
const startUrl = process.env.OPTICHAT_START_URL || 'http://127.0.0.1:3000';
const workspaceRoot = path.resolve(__dirname, '..', '..');
const runtimeDefaultsPath = path.join(workspaceRoot, 'config', 'runtime_defaults.json');

function loadRuntimeDefaults() {
  const fallback = {
    gpuBudgetMb: 6 * 1024,
    fastSingleInstanceVramMb: 91,
    thinkingSingleInstanceVramMb: 123,
  };

  try {
    const raw = JSON.parse(fs.readFileSync(runtimeDefaultsPath, 'utf8'));
    const gpuBudgetMb = Number.parseInt(String(raw.gpu_budget_mb), 10);
    const fastSingleInstanceVramMb = Number.parseInt(String(raw.fast_single_instance_vram_mb), 10);
    const thinkingSingleInstanceVramMb = Number.parseInt(String(raw.thinking_single_instance_vram_mb), 10);

    if ([gpuBudgetMb, fastSingleInstanceVramMb, thinkingSingleInstanceVramMb].some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new Error('Invalid runtime defaults config');
    }

    return {
      gpuBudgetMb,
      fastSingleInstanceVramMb,
      thinkingSingleInstanceVramMb,
    };
  } catch {
    return fallback;
  }
}

const runtimeDefaults = loadRuntimeDefaults();
const GPU_BUDGET_MB = runtimeDefaults.gpuBudgetMb;
const FAST_SINGLE_INSTANCE_VRAM_MB = runtimeDefaults.fastSingleInstanceVramMb;
const THINKING_SINGLE_INSTANCE_VRAM_MB = runtimeDefaults.thinkingSingleInstanceVramMb;
const FAST_DEFAULT_INSTANCE_PARALLELISM = Math.max(1, Math.floor(GPU_BUDGET_MB / FAST_SINGLE_INSTANCE_VRAM_MB));
const THINKING_DEFAULT_INSTANCE_PARALLELISM = Math.max(1, Math.floor(GPU_BUDGET_MB / THINKING_SINGLE_INSTANCE_VRAM_MB));
const DEFAULT_LOOKAHEAD_CHUNK_SIZE = 256;

const defaultSettings = Object.freeze({
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  openaiModel: 'gpt-5.4',
  drlSamples: 128,
  enableLookahead: true,
  lookaheadTopK: 3,
  lookaheadConfidentProb: 0.95,
  enableLocalSearch: true,
  fastLocalSearchRounds: 8,
  thinkingLocalSearchRounds: 50,
});

const ALLOWED_TOOL_PLAN_STEPS = new Set([
  'construct_initial',
  'validate_solution',
  'reduce_vehicles',
  'apply_lookahead',
  'elite_guided_repair',
  'destroy_repair',
  'improve_solution',
  'compare_solutions',
]);

const ALLOWED_VRP_OPERATORS = new Set([
  'two_opt',
  'relocate',
  'swap',
  'or_opt',
  'two_opt_star',
  'cross_exchange',
  'route_elimination',
  'shaw_regret',
]);

const pendingIngestTasks = new Map();

const agentDecisionPrompt = [
  'You are the orchestration agent for a desktop routing assistant.',
  'Decide whether the local solve skill should be called, and optionally attach a solver strategy.',
  'Return JSON only. No markdown, no code fences, no commentary.',
  'If the user mentions penalties, priorities, or tradeoffs such as minimizing vehicles first, paying fixed cost per vehicle, penalizing lateness, or accepting longer distance for fewer vehicles, extract them into solver_config.objective.',
  'Examples of objective extraction:',
  '- "优先少用车，距离长一点也可以" -> {"primary":"vehicle_count"}',
  '- "每多开一辆车罚 100" -> {"vehicle_fixed_cost":100}',
  '- "迟到重罚，超时也要罚" -> {"lateness_penalty":..., "overtime_penalty":...}',
  'If the user clearly wants solving, routing, optimization, improvement, DRL construction, lookahead, local search, or wants to solve the uploaded instance, return one of:',
  '1. {"action":"solve","payload":{...}}',
  '2. {"action":"solve","use_uploaded_payload":true}',
  'You may also attach "solver_config" when returning solve. Example:',
  '3. {"action":"solve","use_uploaded_payload":true,"solver_config":{"tool_plan":["construct_initial","validate_solution","reduce_vehicles","apply_lookahead","improve_solution","compare_solutions"],"enable_vehicle_reduction":true,"local_search_operators":["or_opt","relocate","swap","two_opt"],"objective":{"primary":"vehicle_count","vehicle_fixed_cost":100.0}}}',
  'If the user is asking a general question, wants explanation, or is chatting, return:',
  '4. {"action":"reply","message":"..."}',
  'Never choose solve unless the intent is clearly to solve or optimize.',
  'When you return a payload, the top-level object must contain "problem_type" and "instance".',
  'Supported payloads:',
  'TSP: {"problem_type":"tsp","instance":{"points":[[x,y],...]}}',
  'CVRP: {"problem_type":"cvrp","instance":{"depot_xy":[x,y],"node_xy":[[x,y],...],"node_demand":[...],"capacity":number}}',
  'CVRPTW: {"problem_type":"cvrptw","instance":{"depot_xy":[x,y],"node_xy":[[x,y],...],"node_demand":[...],"capacity":number,"node_tw":[[start,end],...],"service_time":number|[...],"grid_scale":number}}',
  'If information is insufficient for solving, prefer {"action":"reply","message":"..."} and explain what is missing.',
].join('\n');

const solverStrategyPrompt = [
  'You choose a local tool plan for a routing solver orchestrator.',
  'Return JSON only. No markdown.',
  'Format: {"solver_config":{...}}',
  'Read the user tradeoffs carefully and convert them into solver_config.objective when relevant.',
  'If the user states penalties in natural language, convert them into explicit numeric or structured objective fields whenever possible.',
  'Allowed tool_plan steps: construct_initial, validate_solution, reduce_vehicles, apply_lookahead, elite_guided_repair, destroy_repair, improve_solution, compare_solutions.',
  'Allowed operators: two_opt, relocate, swap, or_opt, two_opt_star, cross_exchange, route_elimination, shaw_regret.',
  'You may also set solver_config.objective, for example {"primary":"vehicle_count","vehicle_fixed_cost":100.0,"lateness_penalty":10.0}.',
  'Every plan must start with construct_initial and end with compare_solutions.',
  'Insert validate_solution after each major action.',
  'For TSP never use reduce_vehicles or route_elimination.',
  'For CVRP and CVRPTW you may use reduce_vehicles and route_elimination.',
  'Prefer compact plans. Only enable expensive steps when they are likely to help.',
  'If the user explicitly wants speed, keep the plan short.',
].join('\n');

const refinementStrategyPrompt = [
  'You choose a post-lookahead refinement strategy for a routing solver.',
  'Return JSON only. No markdown.',
  'Format: {"refinement_strategy":{...}}',
  'This stage starts from an existing feasible solution after lookahead.',
  'You may enable elite-guided repair right after lookahead to mine common structure from elite DRL candidates and repair from that consensus start.',
  'Extract user-stated penalties and priorities into objective when needed, but preserve the current objective unless there is a strong reason to change it.',
  'Use the supplied analysis to pick operators. For example, long waiting suggests relocate/or_opt/cross_exchange; too many routes suggests route_elimination.',
  'Pay special attention to hotspots such as longest_edges and worst_connection_points. Large depot-connected anomalies or large excess_over_knn usually suggest relocate, or_opt, cross_exchange, or destroy-repair.',
  'Allowed operators: two_opt, relocate, swap, or_opt, two_opt_star, cross_exchange, route_elimination, shaw_regret.',
  'Choose local_search_operators, destroy_repair_operators, local_search_rounds, destroy_repair_rounds, improvement_cycles, enable_destroy_repair, enable_vehicle_reduction, and enable_elite_guided_repair.',
  'You may also set allow_worse_acceptance, acceptance_budget, acceptance_temperature, acceptance_decay, granular_neighbor_k, regret_k, shaw_remove_count, elite_guided_pool_size, elite_guided_polish_rounds, and elite_guided_candidate_count.',
  'Do not treat elite-guided repair as only a boolean toggle. When it is enabled, explicitly tune elite_guided_pool_size, elite_guided_polish_rounds, and elite_guided_candidate_count from the hotspot pattern.',
  'Heuristic guidance: if hotspots are concentrated on a few routes or a few very bad depot-connected points, use a smaller or medium elite_guided_pool_size (8-24) and moderate polish rounds (6-12).',
  'Heuristic guidance: if hotspots are diffuse across many routes, many long edges compete, or the structure looks globally unstable, use a larger elite_guided_pool_size (24-48), larger elite_guided_candidate_count (64-128), and more polish rounds (12-24).',
  'If the current solution already looks smooth and hotspot severity is low, keep enable_elite_guided_repair false or use very conservative values.',
  'elite_guided_candidate_count should usually stay <= drl_samples. elite_guided_pool_size should usually stay <= elite_guided_candidate_count.',
  'Prefer multiple refinement passes for thinking mode.',
  'Use route_elimination only for CVRP or CVRPTW.',
  'If the analysis shows long waiting or route imbalance, pick operators that address it.',
].join('\n');

const conversationTitlePrompt = [
  'You generate short Chinese chat titles for a sidebar.',
  'Return JSON only. No markdown.',
  'Format: {"title":"..."}',
  'The title should be concise, specific, and usually within 4 to 18 Chinese characters.',
  'Do not use quotes, numbering, emojis, or trailing punctuation.',
  'Prefer concrete task names such as "CVRP TSP 标准格式" or "VRPTW 求解方法".',
].join('\n');

function createPythonCandidates(moduleName) {
  return [
    { command: 'python', args: ['-m', moduleName] },
    { command: 'py', args: ['-3', '-m', moduleName] },
  ];
}

function sanitizeFileName(name, fallback = 'solve-result.json') {
  const normalized = String(name || '').trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g, '-');
  return normalized || fallback;
}

function resolveSettingsFile() {
  return path.join(app.getPath('userData'), 'app-settings.json');
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function sanitizeSettings(input) {
  const source = input && typeof input === 'object' ? input : {};
  const legacyLocalSearchRounds = source.localSearchRounds;
  const legacyLookaheadTopK = source.lookaheadTopK ?? source.lookaheadDepth;
  const lookaheadConfidentProb = Number(source.lookaheadConfidentProb);
  const requestedModel =
    typeof source.openaiModel === 'string' && source.openaiModel.trim()
      ? source.openaiModel.trim().slice(0, 256)
      : defaultSettings.openaiModel;
  return {
    openaiBaseUrl:
      typeof source.openaiBaseUrl === 'string' && source.openaiBaseUrl.trim()
        ? source.openaiBaseUrl.trim()
        : defaultSettings.openaiBaseUrl,
    openaiApiKey: typeof source.openaiApiKey === 'string' ? source.openaiApiKey.trim().slice(0, 4096) : '',
    openaiModel: requestedModel === 'gpt-5.4-mini' ? 'gpt-5.4' : requestedModel,
    drlSamples: clampInt(source.drlSamples, 1, 2048, defaultSettings.drlSamples),
    enableLookahead: typeof source.enableLookahead === 'boolean' ? source.enableLookahead : defaultSettings.enableLookahead,
    lookaheadTopK: clampInt(legacyLookaheadTopK, 1, 32, defaultSettings.lookaheadTopK),
    lookaheadConfidentProb: Number.isFinite(lookaheadConfidentProb)
      ? Math.max(0.5, Math.min(0.999, lookaheadConfidentProb))
      : defaultSettings.lookaheadConfidentProb,
    enableLocalSearch: typeof source.enableLocalSearch === 'boolean' ? source.enableLocalSearch : defaultSettings.enableLocalSearch,
    fastLocalSearchRounds: clampInt(
      source.fastLocalSearchRounds ?? legacyLocalSearchRounds,
      1,
      1000,
      defaultSettings.fastLocalSearchRounds,
    ),
    thinkingLocalSearchRounds: clampInt(
      source.thinkingLocalSearchRounds ?? legacyLocalSearchRounds,
      1,
      1000,
      defaultSettings.thinkingLocalSearchRounds,
    ),
  };
}

function readStoredApiKey(raw) {
  if (!raw || typeof raw !== 'object') {
    return '';
  }

  if (raw.openaiApiKeySecure && safeStorage && safeStorage.isEncryptionAvailable()) {
    try {
      return safeStorage.decryptString(Buffer.from(raw.openaiApiKeySecure, 'base64'));
    } catch {
      return '';
    }
  }

  return typeof raw.openaiApiKey === 'string' ? raw.openaiApiKey : '';
}

function buildPublicSettings(settings, storedApiKey) {
  return {
    ...settings,
    openaiApiKey: '',
    hasStoredOpenaiApiKey: Boolean(storedApiKey),
    openaiApiKeyLast4: storedApiKey ? storedApiKey.slice(-4) : '',
  };
}

function loadSettingsWithSecret() {
  const filePath = resolveSettingsFile();
  if (!fs.existsSync(filePath)) {
    return { ...defaultSettings };
  }

  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const normalized = sanitizeSettings(raw);
    normalized.openaiApiKey = readStoredApiKey(raw);
    return normalized;
  } catch {
    return { ...defaultSettings };
  }
}

function loadSettings() {
  const settings = loadSettingsWithSecret();
  return buildPublicSettings(settings, settings.openaiApiKey);
}

function saveSettings(input) {
  const normalized = sanitizeSettings(input);
  const existing = loadSettingsWithSecret();
  const nextApiKey = normalized.openaiApiKey || existing.openaiApiKey || '';
  const filePath = resolveSettingsFile();
  const persisted = {
    ...normalized,
    openaiApiKey: '',
  };

  if (safeStorage && safeStorage.isEncryptionAvailable() && nextApiKey) {
    const encrypted = safeStorage.encryptString(nextApiKey);
    persisted.openaiApiKeySecure = encrypted.toString('base64');
    delete persisted.openaiApiKey;
  } else if (nextApiKey) {
    persisted.openaiApiKey = nextApiKey;
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(persisted, null, 2), 'utf8');
  return buildPublicSettings(normalized, nextApiKey);
}

function resolveEffectiveSettings(input) {
  const requested = sanitizeSettings(input);
  const stored = loadSettingsWithSecret();
  return {
    ...stored,
    ...requested,
    openaiApiKey: requested.openaiApiKey || stored.openaiApiKey || '',
  };
}

function isJsonLike(text) {
  const trimmed = String(text || '').trim();
  return trimmed.startsWith('{') || trimmed.startsWith('[');
}

function extractStructuredText(content) {
  if (typeof content !== 'string') {
    return '';
  }

  const fenced = content.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    return fenced[1].trim();
  }
  return content.trim();
}

function parseJsonObject(text) {
  const parsed = JSON.parse(text);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('返回内容必须是 JSON 对象。');
  }
  return parsed;
}

function parseJsonPayload(text) {
  const parsed = parseJsonObject(text);
  const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : parsed;
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('输入的 JSON 必须是对象。');
  }
  if (payload.error) {
    throw new Error(typeof payload.error === 'string' ? payload.error : '模型没有生成有效的 payload。');
  }
  if (!payload.problem_type || !payload.instance || typeof payload.instance !== 'object') {
    throw new Error('payload 缺少 problem_type 或 instance。');
  }
  return payload;
}

function resolveResponsesUrl(baseUrl) {
  const trimmed = String(baseUrl || '').trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'https://api.openai.com/v1/responses';
  }
  if (trimmed.endsWith('/responses')) {
    return trimmed;
  }
  if (trimmed.endsWith('/chat/completions')) {
    return `${trimmed.slice(0, -'/chat/completions'.length)}/responses`;
  }
  return `${trimmed}/responses`;
}

function buildReasoningConfig(settings) {
  const model = String(settings?.openaiModel || '')
    .trim()
    .toLowerCase();
  if (!model.startsWith('gpt-5')) {
    return null;
  }
  return { effort: 'xhigh' };
}

function buildOpenAIRequestBody(settings, body) {
  const reasoning = buildReasoningConfig(settings);
  return reasoning ? { ...body, reasoning } : body;
}

function getResponseText(responsePayload) {
  if (typeof responsePayload?.output_text === 'string' && responsePayload.output_text.trim()) {
    return responsePayload.output_text.trim();
  }

  if (!Array.isArray(responsePayload?.output)) {
    return '';
  }

  return responsePayload.output
    .flatMap((item) => (Array.isArray(item?.content) ? item.content : []))
    .map((part) => (part && typeof part.text === 'string' ? part.text : ''))
    .join('')
    .trim();
}

function redactSecrets(value) {
  return String(value || '')
    .replace(/sk-[A-Za-z0-9_*.-]+/g, 'sk-[redacted]')
    .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, 'Bearer [redacted]');
}

function sanitizeSuggestedTitle(title, fallback = '新对话') {
  const normalized = String(title || '')
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/^["'“”]+|["'“”]+$/g, '')
    .replace(/[。！？!?，、；;:：]+$/g, '');
  if (!normalized) {
    return fallback;
  }
  return normalized.length > 24 ? normalized.slice(0, 24) : normalized;
}

function buildFallbackConversationTitle({ text, uploadedFiles, ingestResults, payload }) {
  const normalizedText = String(text || '').replace(/\s+/g, ' ').trim();
  if (normalizedText) {
    return sanitizeSuggestedTitle(normalizedText);
  }

  if (Array.isArray(uploadedFiles) && uploadedFiles.length > 0) {
    if (uploadedFiles.length === 1) {
      return sanitizeSuggestedTitle(path.basename(uploadedFiles[0].name, path.extname(uploadedFiles[0].name)));
    }
    return sanitizeSuggestedTitle(`批量求解 ${uploadedFiles.length} 个实例`);
  }

  const firstSummary = Array.isArray(ingestResults) ? ingestResults[0]?.summary : null;
  const problemType = payload?.problem_type || firstSummary?.problem_type;
  if (problemType) {
    return sanitizeSuggestedTitle(`${String(problemType).toUpperCase()} 求解`);
  }

  return '新对话';
}

function buildHtmlResponseError(url, contentType, responseText) {
  const snippet = redactSecrets(String(responseText || '').replace(/\s+/g, ' ').slice(0, 160));
  return new Error(
    `OpenAI 接口返回了 HTML 页面，而不是 JSON。通常是 Base URL 配错了，或代理/登录页拦截了请求。\n请求地址：${url}\nContent-Type：${
      contentType || 'unknown'
    }\n响应片段：${snippet}`,
  );
}

function parseJsonHttpResponse(url, response, responseText) {
  const contentType = response.headers.get('content-type') || '';
  const trimmed = redactSecrets(String(responseText || '').trim());
  const lower = trimmed.toLowerCase();
  const looksHtml =
    contentType.includes('text/html') ||
    lower.startsWith('<!doctype') ||
    lower.startsWith('<html') ||
    trimmed.startsWith('<');

  if (!response.ok) {
    if (looksHtml) {
      throw buildHtmlResponseError(url, contentType, responseText);
    }
    throw new Error(`OpenAI 请求失败：${response.status} ${response.statusText}\n${redactSecrets(String(responseText || '').slice(0, 600))}`);
  }

  if (looksHtml) {
    throw buildHtmlResponseError(url, contentType, responseText);
  }

  try {
    return JSON.parse(responseText);
  } catch {
    throw new Error(
      `OpenAI 接口返回的不是合法 JSON。通常是 Base URL 指向了网页或非兼容接口。\n请求地址：${url}\n响应片段：${trimmed.slice(0, 160)}`,
    );
  }
}

async function postOpenAIResponse(settings, body) {
  const url = resolveResponsesUrl(settings.openaiBaseUrl);
  const send = async (requestBody) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${settings.openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    const responseText = await response.text();
    return { response, responseText };
  };

  const requestBody = buildOpenAIRequestBody(settings, body);
  let { response, responseText } = await send(requestBody);
  try {
    return parseJsonHttpResponse(url, response, responseText);
  } catch (error) {
    const retryWithoutReasoning =
      requestBody.reasoning &&
      response.status === 400 &&
      /reasoning|unsupported|unknown parameter/i.test(String(responseText || ''));
    if (!retryWithoutReasoning) {
      throw error;
    }
    ({ response, responseText } = await send(body));
    return parseJsonHttpResponse(url, response, responseText);
  }
}

function emitProgress(sender, requestId, stepId, label, status, detail = null) {
  sender.send('solver:progress', {
    requestId,
    stepId,
    label,
    status,
    detail,
    timestamp: new Date().toISOString(),
  });
}

function buildRuntimeDefaultsPayload() {
  return {
    gpuBudgetMb: GPU_BUDGET_MB,
    fastSingleInstanceVramMb: FAST_SINGLE_INSTANCE_VRAM_MB,
    thinkingSingleInstanceVramMb: THINKING_SINGLE_INSTANCE_VRAM_MB,
    fastMaxParallelInstances: FAST_DEFAULT_INSTANCE_PARALLELISM,
    thinkingMaxParallelInstances: THINKING_DEFAULT_INSTANCE_PARALLELISM,
  };
}

function normalizeOperatorList(value) {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const normalized = value
    .map((item) => String(item || '').trim().toLowerCase())
    .filter((item) => ALLOWED_VRP_OPERATORS.has(item));
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeObjectiveSpec(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }

  const source = value.objective && typeof value.objective === 'object' ? value.objective : value;
  const normalized = {};
  const primary = String(source.primary || '').trim().toLowerCase();
  if (primary === 'distance' || primary === 'vehicle_count') {
    normalized.primary = primary;
  }
  if (typeof source.prioritize_vehicle_count === 'boolean') {
    normalized.prioritize_vehicle_count = source.prioritize_vehicle_count;
  }

  const numericFields = [
    'distance_weight',
    'duration_weight',
    'vehicle_fixed_cost',
    'vehicle_count_weight',
    'overtime_penalty',
    'lateness_penalty',
    'unserved_penalty',
  ];
  for (const field of numericFields) {
    if (source[field] !== undefined && source[field] !== null && source[field] !== '') {
      const parsed = Number(source[field]);
      if (Number.isFinite(parsed)) {
        normalized[field] = parsed;
      }
    }
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function normalizeToolPlan(value, quickMode) {
  if (quickMode) {
    return ['construct_initial', 'validate_solution', 'improve_solution', 'validate_solution', 'compare_solutions'];
  }
  if (!Array.isArray(value)) {
    return undefined;
  }

  const normalized = value
    .map((item) => String(item || '').trim())
    .filter((item) => ALLOWED_TOOL_PLAN_STEPS.has(item));

  if (normalized.length === 0) {
    return undefined;
  }
  if (normalized[0] !== 'construct_initial' || normalized[normalized.length - 1] !== 'compare_solutions') {
    return undefined;
  }
  return normalized;
}

function buildThinkingWarmupToolPlan() {
  return ['construct_initial', 'validate_solution', 'compare_solutions'];
}

function normalizeSolverConfig(value, quickMode) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const normalized = {};
  const toolPlan = normalizeToolPlan(value.tool_plan, quickMode);
  if (toolPlan) {
    normalized.tool_plan = toolPlan;
  }

  const lookaheadOperators = normalizeOperatorList(value.lookahead_operators);
  if (lookaheadOperators) {
    normalized.lookahead_operators = lookaheadOperators;
  }

  const localSearchOperators = normalizeOperatorList(value.local_search_operators);
  if (localSearchOperators) {
    normalized.local_search_operators = localSearchOperators;
  }

  const destroyRepairOperators = normalizeOperatorList(value.destroy_repair_operators);
  if (destroyRepairOperators) {
    normalized.destroy_repair_operators = destroyRepairOperators;
  }

  const objective = normalizeObjectiveSpec(value.objective ?? value.objective_spec);
  if (objective) {
    normalized.objective = objective;
  }

  if (!quickMode) {
    if (typeof value.enable_vehicle_reduction === 'boolean') {
      normalized.enable_vehicle_reduction = value.enable_vehicle_reduction;
    }
    if (typeof value.enable_lookahead === 'boolean') {
      normalized.enable_lookahead = value.enable_lookahead;
    }
    if (typeof value.enable_destroy_repair === 'boolean') {
      normalized.enable_destroy_repair = value.enable_destroy_repair;
    }
    if (typeof value.enable_local_search === 'boolean') {
      normalized.enable_local_search = value.enable_local_search;
    }
    if (typeof value.enable_elite_guided_repair === 'boolean') {
      normalized.enable_elite_guided_repair = value.enable_elite_guided_repair;
    }
    if (typeof value.decode_lookahead_as_initial === 'boolean') {
      normalized.decode_lookahead_as_initial = value.decode_lookahead_as_initial;
    }
    if (typeof value.allow_worse_acceptance === 'boolean') {
      normalized.allow_worse_acceptance = value.allow_worse_acceptance;
    }
  }

  const integerFields = [
    ['drl_samples', 1, 2048],
    ['seed_trials', 1, 32],
    ['initial_candidate_count', 1, 128],
    ['lookahead_depth', 1, 8],
    ['lookahead_beam_width', 1, 512],
    ['lookahead_top_k', 1, 32],
    ['lookahead_uncertain_chunk_size', 1, 512],
    ['lookahead_per_operator_limit', 1, 64],
    ['local_search_rounds', 1, 1000],
    ['destroy_repair_rounds', 1, 1000],
    ['vehicle_reduction_rounds', 1, 1000],
    ['granular_neighbor_k', 1, 64],
    ['regret_k', 2, 8],
    ['shaw_remove_count', 2, 64],
    ['elite_guided_pool_size', 2, 64],
    ['elite_guided_polish_rounds', 0, 128],
    ['elite_guided_seed_trials', 1, 16],
    ['elite_guided_candidate_count', 2, 256],
    ['acceptance_budget', 0, 128],
    ['random_seed', 0, 1000000],
  ];

  for (const [field, min, max] of integerFields) {
    if (value[field] !== undefined && value[field] !== null && value[field] !== '') {
      normalized[field] = clampInt(value[field], min, max, min);
    }
  }

  if (value.lookahead_confident_prob !== undefined && value.lookahead_confident_prob !== null && value.lookahead_confident_prob !== '') {
    normalized.lookahead_confident_prob = Math.max(0.5, Math.min(0.999, Number(value.lookahead_confident_prob)));
  }
  if (value.acceptance_temperature !== undefined && value.acceptance_temperature !== null && value.acceptance_temperature !== '') {
    normalized.acceptance_temperature = Math.max(0.0001, Math.min(1.0, Number(value.acceptance_temperature)));
  }
  if (value.acceptance_decay !== undefined && value.acceptance_decay !== null && value.acceptance_decay !== '') {
    normalized.acceptance_decay = Math.max(0.01, Math.min(1.0, Number(value.acceptance_decay)));
  }

  return normalized;
}

function normalizeRefinementStrategy(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const source =
    value.refinement_strategy && typeof value.refinement_strategy === 'object' ? value.refinement_strategy : value;
  const normalized = {};
  const localSearchOperators = normalizeOperatorList(source.local_search_operators);
  if (localSearchOperators) {
    normalized.local_search_operators = localSearchOperators;
  }
  const destroyRepairOperators = normalizeOperatorList(source.destroy_repair_operators);
  if (destroyRepairOperators) {
    normalized.destroy_repair_operators = destroyRepairOperators;
  }
  const objective = normalizeObjectiveSpec(source.objective ?? source.objective_spec);
  if (objective) {
    normalized.objective = objective;
  }
  if (typeof source.enable_destroy_repair === 'boolean') {
    normalized.enable_destroy_repair = source.enable_destroy_repair;
  }
  if (typeof source.enable_vehicle_reduction === 'boolean') {
    normalized.enable_vehicle_reduction = source.enable_vehicle_reduction;
  }
  if (typeof source.enable_elite_guided_repair === 'boolean') {
    normalized.enable_elite_guided_repair = source.enable_elite_guided_repair;
  }
  if (typeof source.allow_worse_acceptance === 'boolean') {
    normalized.allow_worse_acceptance = source.allow_worse_acceptance;
  }

  const integerFields = [
    ['improvement_cycles', 1, 6],
    ['local_search_rounds', 1, 1000],
    ['destroy_repair_rounds', 1, 1000],
    ['vehicle_reduction_rounds', 1, 1000],
    ['granular_neighbor_k', 1, 64],
    ['regret_k', 2, 8],
    ['shaw_remove_count', 2, 64],
    ['elite_guided_pool_size', 2, 64],
    ['elite_guided_polish_rounds', 0, 128],
    ['elite_guided_seed_trials', 1, 16],
    ['elite_guided_candidate_count', 2, 256],
    ['acceptance_budget', 0, 128],
    ['random_seed', 0, 1000000],
  ];
  for (const [field, min, max] of integerFields) {
    if (source[field] !== undefined && source[field] !== null && source[field] !== '') {
      normalized[field] = clampInt(source[field], min, max, min);
    }
  }
  if (source.acceptance_temperature !== undefined && source.acceptance_temperature !== null && source.acceptance_temperature !== '') {
    normalized.acceptance_temperature = Math.max(0.0001, Math.min(1.0, Number(source.acceptance_temperature)));
  }
  if (source.acceptance_decay !== undefined && source.acceptance_decay !== null && source.acceptance_decay !== '') {
    normalized.acceptance_decay = Math.max(0.01, Math.min(1.0, Number(source.acceptance_decay)));
  }
  return normalized;
}

function withSolveDefaults(payload, mode, settings, solverConfig = null) {
  const quickMode = mode === 'quick';
  const normalizedSolverConfig = normalizeSolverConfig(solverConfig, quickMode);
  const rawConfig = {
    ...(payload.config && typeof payload.config === 'object' ? payload.config : {}),
    ...normalizedSolverConfig,
  };
  const defaultSeedTrials = quickMode ? 8 : 1;
  const defaultLocalSearchRounds = quickMode ? settings.fastLocalSearchRounds : settings.thinkingLocalSearchRounds;

  return {
    ...payload,
    problem_type: String(payload.problem_type).trim().toLowerCase(),
    config: {
      ...rawConfig,
      mode: quickMode ? 'fast' : 'hybrid',
      drl_samples: clampInt(rawConfig.drl_samples, 1, 2048, settings.drlSamples),
      seed_trials: clampInt(rawConfig.seed_trials, 1, 32, defaultSeedTrials),
      enable_lookahead: quickMode ? false : rawConfig.enable_lookahead ?? settings.enableLookahead,
      lookahead_depth: clampInt(rawConfig.lookahead_depth, 1, 8, settings.lookaheadTopK),
      lookahead_beam_width: clampInt(rawConfig.lookahead_beam_width, 1, 512, DEFAULT_LOOKAHEAD_CHUNK_SIZE),
      lookahead_top_k: clampInt(rawConfig.lookahead_top_k, 1, 32, settings.lookaheadTopK),
      lookahead_confident_prob:
        rawConfig.lookahead_confident_prob !== undefined && rawConfig.lookahead_confident_prob !== null && rawConfig.lookahead_confident_prob !== ''
          ? Math.max(0.5, Math.min(0.999, Number(rawConfig.lookahead_confident_prob)))
          : settings.lookaheadConfidentProb,
      lookahead_uncertain_chunk_size: clampInt(
        rawConfig.lookahead_uncertain_chunk_size,
        1,
        2048,
        DEFAULT_LOOKAHEAD_CHUNK_SIZE,
      ),
      lookahead_k: clampInt(rawConfig.lookahead_k, 1, 64, 1),
      enable_local_search: quickMode ? true : rawConfig.enable_local_search ?? true,
      local_search_rounds: clampInt(rawConfig.local_search_rounds, 1, 1000, defaultLocalSearchRounds),
      granular_neighbor_k: clampInt(rawConfig.granular_neighbor_k, 1, 64, quickMode ? 16 : 24),
      regret_k: clampInt(rawConfig.regret_k, 2, 8, 3),
      shaw_remove_count: clampInt(rawConfig.shaw_remove_count, 2, 64, quickMode ? 4 : 8),
      enable_elite_guided_repair: quickMode ? false : rawConfig.enable_elite_guided_repair ?? false,
      elite_guided_pool_size: clampInt(rawConfig.elite_guided_pool_size, 2, 64, 24),
      elite_guided_polish_rounds: clampInt(rawConfig.elite_guided_polish_rounds, 0, 128, 12),
      elite_guided_seed_trials: clampInt(rawConfig.elite_guided_seed_trials, 1, 16, 1),
      elite_guided_candidate_count: clampInt(rawConfig.elite_guided_candidate_count, 2, 256, settings.drlSamples),
      allow_worse_acceptance: quickMode ? false : rawConfig.allow_worse_acceptance ?? true,
      acceptance_budget: clampInt(rawConfig.acceptance_budget, 0, 128, quickMode ? 0 : Math.max(2, Math.floor(defaultLocalSearchRounds / 6))),
      acceptance_temperature:
        rawConfig.acceptance_temperature !== undefined && rawConfig.acceptance_temperature !== null && rawConfig.acceptance_temperature !== ''
          ? Math.max(0.0001, Math.min(1.0, Number(rawConfig.acceptance_temperature)))
          : 0.01,
      acceptance_decay:
        rawConfig.acceptance_decay !== undefined && rawConfig.acceptance_decay !== null && rawConfig.acceptance_decay !== ''
          ? Math.max(0.01, Math.min(1.0, Number(rawConfig.acceptance_decay)))
          : 0.9,
      random_seed: clampInt(rawConfig.random_seed, 0, 1000000, 0),
      ...(rawConfig.tool_plan ? { tool_plan: rawConfig.tool_plan } : {}),
      ...(rawConfig.lookahead_operators ? { lookahead_operators: rawConfig.lookahead_operators } : {}),
      ...(rawConfig.local_search_operators
        ? { local_search_operators: rawConfig.local_search_operators }
        : quickMode
          ? { local_search_operators: ['two_opt', 'relocate', 'swap'] }
          : {}),
      ...(rawConfig.destroy_repair_operators ? { destroy_repair_operators: rawConfig.destroy_repair_operators } : {}),
      ...(rawConfig.objective ? { objective: rawConfig.objective } : {}),
      ...(rawConfig.enable_vehicle_reduction !== undefined ? { enable_vehicle_reduction: Boolean(rawConfig.enable_vehicle_reduction) } : {}),
      ...(rawConfig.enable_destroy_repair !== undefined ? { enable_destroy_repair: Boolean(rawConfig.enable_destroy_repair) } : {}),
      ...(rawConfig.initial_candidate_count !== undefined
        ? { initial_candidate_count: clampInt(rawConfig.initial_candidate_count, 1, 128, 1) }
        : {}),
      ...(rawConfig.lookahead_per_operator_limit !== undefined
        ? { lookahead_per_operator_limit: clampInt(rawConfig.lookahead_per_operator_limit, 1, 64, DEFAULT_LOOKAHEAD_CHUNK_SIZE) }
        : {}),
      ...(rawConfig.destroy_repair_rounds !== undefined
        ? { destroy_repair_rounds: clampInt(rawConfig.destroy_repair_rounds, 1, 1000, defaultLocalSearchRounds) }
        : {}),
      ...(rawConfig.vehicle_reduction_rounds !== undefined
        ? { vehicle_reduction_rounds: clampInt(rawConfig.vehicle_reduction_rounds, 1, 1000, 10) }
        : {}),
    },
  };
}

function runJsonModuleWithCandidate(moduleName, candidate, options = {}) {
  const { args = [], stdinPayload = null } = options;

  return new Promise((resolve, reject) => {
    const child = spawn(candidate.command, [...candidate.args, ...args], {
      cwd: workspaceRoot,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `${candidate.command} exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        const message = error instanceof Error ? error.message : 'unknown error';
        reject(new Error(`${moduleName} 返回了无效 JSON：${message}\n${stdout.slice(0, 600)}`));
      }
    });

    if (stdinPayload !== null) {
      child.stdin.write(JSON.stringify(stdinPayload));
    }
    child.stdin.end();
  });
}

async function runJsonModule(moduleName, options = {}) {
  const startedAt = Date.now();
  const errors = [];

  for (const candidate of createPythonCandidates(moduleName)) {
    try {
      const result = await runJsonModuleWithCandidate(moduleName, candidate, options);
      return {
        result,
        durationMs: Date.now() - startedAt,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${candidate.command}: ${message}`);
    }
  }

  throw new Error(`无法调用本地 Python 模块 ${moduleName}。\n${errors.join('\n\n')}`);
}

function createCancellationError() {
  const error = new Error('INGEST_CANCELLED');
  error.code = 'INGEST_CANCELLED';
  return error;
}

function runInstanceIngestCandidate(task, candidate, filePath) {
  return new Promise((resolve, reject) => {
    if (task.cancelled) {
      reject(createCancellationError());
      return;
    }

    const child = spawn(candidate.command, [...candidate.args, '--input-file', filePath], {
      cwd: workspaceRoot,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    task.child = child;
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      if (task.cancelled) {
        reject(createCancellationError());
        return;
      }

      if (code !== 0) {
        reject(new Error(stderr.trim() || `${candidate.command} exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        const message = error instanceof Error ? error.message : 'unknown error';
        reject(new Error(`instance_skill.cli 杩斿洖浜嗘棤鏁?JSON锛?{message}\n${stdout.slice(0, 600)}`));
      }
    });
  });
}

async function runCancelableInstanceIngest(requestId, filePath) {
  const taskId = typeof requestId === 'string' && requestId.trim() ? requestId.trim() : `ingest-${Date.now()}`;
  const task = { cancelled: false, child: null };
  pendingIngestTasks.set(taskId, task);

  try {
    const errors = [];
    for (const candidate of createPythonCandidates('instance_skill.cli')) {
      if (task.cancelled) {
        throw createCancellationError();
      }

      try {
        return await runInstanceIngestCandidate(task, candidate, filePath);
      } catch (error) {
        if (task.cancelled || error?.code === 'INGEST_CANCELLED' || String(error?.message || '') === 'INGEST_CANCELLED') {
          throw createCancellationError();
        }
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`${candidate.command}: ${message}`);
      }
    }

    throw new Error(`鏃犳硶璋冪敤鏈湴 Python 妯″潡 instance_skill.cli銆俓n${errors.join('\n\n')}`);
  } finally {
    pendingIngestTasks.delete(taskId);
  }
}

function cancelInstanceIngest(requestId) {
  const taskId = typeof requestId === 'string' ? requestId.trim() : '';
  if (!taskId) {
    return false;
  }

  const task = pendingIngestTasks.get(taskId);
  if (!task) {
    return false;
  }

  task.cancelled = true;
  if (task.child && !task.child.killed) {
    try {
      task.child.kill();
    } catch {}
  }
  return true;
}

async function runInstanceIngest(filePath) {
  const execution = await runJsonModule('instance_skill.cli', {
    args: ['--input-file', filePath],
  });
  return execution.result;
}

async function runSolutionAnalysis(problemType, instance, solution, objective = null) {
  const execution = await runJsonModule('tools.analyze_solution', {
    stdinPayload: {
      problem_type: problemType,
      instance,
      solution,
      objective,
    },
  });
  return execution.result;
}

function normalizeUploadedFiles(request) {
  const normalized = [];
  const pushFile = (file) => {
    if (!file || typeof file.path !== 'string') {
      return;
    }
    const item = {
      path: file.path,
      name: typeof file.name === 'string' ? file.name : path.basename(file.path),
    };
    if (!normalized.some((existing) => existing.path === item.path)) {
      normalized.push(item);
    }
  };

  if (Array.isArray(request?.uploadedFiles)) {
    for (const file of request.uploadedFiles) {
      pushFile(file);
    }
  }
  pushFile(request?.uploadedFile);
  return normalized;
}

function mapSolverProgressStep(stepId, label) {
  if (stepId === 'seed_done') {
    return { stepId: 'seed', label: 'DRL 初始解完成', status: 'completed' };
  }
  if (stepId === 'lookahead_done') {
    return { stepId: 'lookahead', label: 'Lookahead 完成', status: 'completed' };
  }
  if (stepId === 'local_search_done') {
    return { stepId: 'local_search', label: '局部搜索完成', status: 'completed' };
  }
  if (stepId === 'finalize') {
    return { stepId: 'finalize', label: '整理最终结果', status: 'completed' };
  }
  return { stepId, label, status: 'running' };
}

function runSolverCandidateWithProgress(sender, requestId, candidate, payload, options = {}) {
  const emitInnerProgress = options.emitInnerProgress !== false;

  return new Promise((resolve, reject) => {
    const child = spawn(candidate.command, [...candidate.args], {
      cwd: workspaceRoot,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderrBuffer = '';
    const stderrLines = [];

    const consumeStderr = (flush = false) => {
      const lines = stderrBuffer.split(/\r?\n/);
      if (!flush) {
        stderrBuffer = lines.pop() ?? '';
      } else {
        stderrBuffer = '';
      }

      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }
        if (line.startsWith('PROGRESS\t')) {
          try {
            const data = JSON.parse(line.slice('PROGRESS\t'.length));
            const mapped = mapSolverProgressStep(data.stepId, data.label);
            if (emitInnerProgress) {
              emitProgress(sender, requestId, mapped.stepId, mapped.label, mapped.status, data.detail ?? null);
            }
          } catch {
            stderrLines.push(line);
          }
        } else {
          stderrLines.push(line);
        }
      }
    };

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString();
      consumeStderr(false);
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      consumeStderr(true);
      if (code !== 0) {
        reject(new Error(stderrLines.join('\n').trim() || `${candidate.command} exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        const message = error instanceof Error ? error.message : 'unknown error';
        reject(new Error(`solver_skill.cli 返回了无效 JSON：${message}\n${stdout.slice(0, 600)}`));
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

async function runSolverWithProgress(sender, requestId, payload, options = {}) {
  const startedAt = Date.now();
  const errors = [];

  for (const candidate of createPythonCandidates('solver_skill.cli')) {
    try {
      const result = await runSolverCandidateWithProgress(sender, requestId, candidate, payload, options);
      return {
        result,
        durationMs: Date.now() - startedAt,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${candidate.command}: ${message}`);
    }
  }

  throw new Error(`无法调用本地 Python 求解器。\n${errors.join('\n\n')}`);
}

function getBatchParallelism(mode, instanceCount) {
  const maxParallelism = mode === 'thinking' ? THINKING_DEFAULT_INSTANCE_PARALLELISM : FAST_DEFAULT_INSTANCE_PARALLELISM;
  return Math.max(1, Math.min(instanceCount, maxParallelism));
}

async function runBatchSolversWithProgress(sender, requestId, batchEntries, mode, settings, structuredText, solverConfig = null) {
  const parallelism = getBatchParallelism(mode, batchEntries.length);
  const batchItems = new Array(batchEntries.length);
  let cursor = 0;
  let completed = 0;

  emitProgress(sender, requestId, 'solve', '批量并行求解', 'running', `共 ${batchEntries.length} 个实例，最大并行 ${parallelism}`);

  const worker = async () => {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= batchEntries.length) {
        return;
      }

      const entry = batchEntries[index];
      const normalizedPayload = withSolveDefaults(entry.ingestResult.payload, mode, settings, solverConfig);
      emitProgress(sender, requestId, 'solve', '批量并行求解', 'running', `开始 ${index + 1}/${batchEntries.length} · ${entry.fileName}`);

      try {
        const solveExecution = await runSolverWithProgress(sender, requestId, normalizedPayload, { emitInnerProgress: false });
        batchItems[index] = {
          fileName: entry.fileName,
          payload: normalizedPayload,
          payloadSource: 'upload',
          result: solveExecution.result,
          durationMs: solveExecution.durationMs,
          structuredText,
          ingestResult: entry.ingestResult,
        };

        completed += 1;
        emitProgress(
          sender,
          requestId,
          'solve',
          '批量并行求解',
          completed === batchEntries.length ? 'completed' : 'running',
          `已完成 ${completed}/${batchEntries.length} · 最新完成 ${entry.fileName}`,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        emitProgress(sender, requestId, 'solve', '批量并行求解', 'error', `${entry.fileName} 失败：${message}`);
        throw error;
      }
    }
  };

  await Promise.all(Array.from({ length: parallelism }, () => worker()));
  return {
    batchItems,
    parallelism,
  };
}

function buildAgentUserPrompt(text, conversation, ingestResult) {
  const context = [];
  if (Array.isArray(conversation) && conversation.length > 0) {
    context.push('Conversation Context:');
    for (const item of conversation) {
      if (item && typeof item.content === 'string' && (item.role === 'user' || item.role === 'assistant')) {
        context.push(`${item.role.toUpperCase()}: ${item.content}`);
      }
    }
  }

  const summaries = Array.isArray(ingestResult) ? ingestResult.map((item) => item?.summary).filter(Boolean) : ingestResult?.summary ? [ingestResult.summary] : [];
  if (summaries.length > 0) {
    context.push('Uploaded Instance Summary:');
    context.push(JSON.stringify(summaries));
  }

  context.push('Current User Message:');
  context.push(text || '(empty)');
  return context.join('\n');
}

function summarizePayloadForStrategy(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return null;
  }

  const instance = payload.instance && typeof payload.instance === 'object' ? payload.instance : {};
  const nodeCount = Array.isArray(instance.points)
    ? instance.points.length
    : Array.isArray(instance.node_xy)
      ? instance.node_xy.length
      : null;

  return {
    problem_type: payload.problem_type ?? null,
    node_count: nodeCount,
    capacity: typeof instance.capacity === 'number' ? instance.capacity : null,
    has_time_windows: Array.isArray(instance.node_tw),
    service_time_kind: Array.isArray(instance.service_time)
      ? 'vector'
      : typeof instance.service_time === 'number'
        ? 'scalar'
        : null,
  };
}

function summarizeRefinementHotspots(analysis) {
  const routes = Array.isArray(analysis?.routes) ? analysis.routes : [];
  const hotspots = analysis?.hotspots && typeof analysis.hotspots === 'object' ? analysis.hotspots : {};
  const longestEdges = Array.isArray(hotspots.longest_edges) ? hotspots.longest_edges : [];
  const worstPoints = Array.isArray(hotspots.worst_connection_points) ? hotspots.worst_connection_points : [];
  const routeCount = routes.length;

  const longestEdgeRoutes = new Set(longestEdges.map((item) => Number(item?.route_index)).filter((value) => Number.isFinite(value)));
  const worstPointRoutes = new Set(worstPoints.map((item) => Number(item?.route_index)).filter((value) => Number.isFinite(value)));
  const affectedRoutes = new Set([...longestEdgeRoutes, ...worstPointRoutes]);

  const maxEdgeDistance = longestEdges.reduce((best, item) => Math.max(best, Number(item?.distance) || 0), 0);
  const maxExcessOverKnn = worstPoints.reduce((best, item) => Math.max(best, Number(item?.excess_over_knn) || 0), 0);
  const waitingTime = Number(analysis?.summary?.max_waiting_time) || 0;

  const concentrated =
    affectedRoutes.size > 0 &&
    ((routeCount > 0 && affectedRoutes.size <= Math.max(1, Math.ceil(routeCount / 3))) || affectedRoutes.size <= 2);

  const diffuse =
    routeCount > 0 &&
    affectedRoutes.size >= Math.max(3, Math.ceil(routeCount * 0.6));

  const severe =
    maxExcessOverKnn >= 20 ||
    maxEdgeDistance >= 20 ||
    waitingTime >= 30;

  return {
    route_count: routeCount,
    longest_edge_count: longestEdges.length,
    worst_connection_point_count: worstPoints.length,
    affected_route_count: affectedRoutes.size,
    max_edge_distance: maxEdgeDistance,
    max_excess_over_knn: maxExcessOverKnn,
    max_waiting_time: waitingTime,
    pattern: diffuse ? 'diffuse' : concentrated ? 'concentrated' : 'mixed',
    severity: severe ? 'high' : maxExcessOverKnn >= 10 || maxEdgeDistance >= 12 || waitingTime >= 15 ? 'medium' : 'low',
    top_longest_edges: longestEdges.slice(0, 3),
    top_worst_connection_points: worstPoints.slice(0, 3),
  };
}

function buildEliteGuidedFallbackStrategy(analysis, baseSolverConfig = {}, settings = defaultSettings) {
  const hotspotSummary = summarizeRefinementHotspots(analysis);
  const drlSamples = Math.max(2, Number(baseSolverConfig?.drl_samples || settings.drlSamples || 128));

  let enableEliteGuidedRepair = false;
  let eliteGuidedPoolSize = 16;
  let eliteGuidedPolishRounds = 8;
  let eliteGuidedCandidateCount = Math.min(64, drlSamples);

  if (hotspotSummary.severity === 'high' && hotspotSummary.pattern === 'diffuse') {
    enableEliteGuidedRepair = true;
    eliteGuidedPoolSize = Math.min(48, drlSamples);
    eliteGuidedPolishRounds = 18;
    eliteGuidedCandidateCount = Math.min(128, drlSamples);
  } else if (hotspotSummary.severity === 'high' && hotspotSummary.pattern === 'concentrated') {
    enableEliteGuidedRepair = true;
    eliteGuidedPoolSize = Math.min(24, drlSamples);
    eliteGuidedPolishRounds = 12;
    eliteGuidedCandidateCount = Math.min(96, drlSamples);
  } else if (hotspotSummary.severity === 'medium' && hotspotSummary.pattern !== 'mixed') {
    enableEliteGuidedRepair = true;
    eliteGuidedPoolSize = Math.min(20, drlSamples);
    eliteGuidedPolishRounds = 10;
    eliteGuidedCandidateCount = Math.min(64, drlSamples);
  }

  return {
    local_search_operators: ['or_opt', 'two_opt_star', 'cross_exchange', 'relocate', 'swap', 'two_opt'],
    destroy_repair_operators: ['shaw_regret', 'route_elimination', 'or_opt', 'relocate', 'swap', 'two_opt'],
    enable_destroy_repair: true,
    enable_vehicle_reduction: false,
    enable_elite_guided_repair: enableEliteGuidedRepair,
    allow_worse_acceptance: true,
    acceptance_budget: 6,
    acceptance_temperature: 0.01,
    acceptance_decay: 0.9,
    granular_neighbor_k: 24,
    regret_k: 3,
    shaw_remove_count: 8,
    elite_guided_pool_size: eliteGuidedPoolSize,
    elite_guided_polish_rounds: eliteGuidedPolishRounds,
    elite_guided_seed_trials: 1,
    elite_guided_candidate_count: eliteGuidedCandidateCount,
  };
}

async function requestSolverStrategy(text, settings, mode, payload, ingestResults) {
  if (!settings.openaiApiKey || mode !== 'thinking') {
    return {};
  }

  try {
    const parsedResponse = await postOpenAIResponse(settings, {
      model: settings.openaiModel,
      store: false,
      instructions: solverStrategyPrompt,
      input: [
        `User request: ${text || '(empty)'}`,
        `Mode: ${mode}`,
        `Payload summary: ${JSON.stringify(summarizePayloadForStrategy(payload))}`,
        Array.isArray(ingestResults) && ingestResults.length > 0
          ? `Uploaded summaries: ${JSON.stringify(ingestResults.map((item) => item.summary).filter(Boolean))}`
          : null,
      ]
        .filter(Boolean)
        .join('\n'),
    });

    const structuredText = extractStructuredText(getResponseText(parsedResponse));
    if (!structuredText) {
      return {};
    }

    const parsed = parseJsonObject(structuredText);
    return normalizeSolverConfig(parsed.solver_config ?? parsed, false);
  } catch {
    return {};
  }
}

function buildRefinementToolPlan(strategy) {
  const cycles = Math.max(1, Number(strategy.improvement_cycles || 2));
  const plan = ['validate_solution'];
  if (strategy.enable_elite_guided_repair) {
    plan.push('elite_guided_repair', 'validate_solution');
  }
  if (strategy.enable_vehicle_reduction) {
    plan.push('reduce_vehicles', 'validate_solution');
  }
  for (let index = 0; index < cycles; index += 1) {
    if (strategy.enable_destroy_repair) {
      plan.push('destroy_repair', 'validate_solution');
    }
    plan.push('improve_solution', 'validate_solution');
  }
  plan.push('compare_solutions');
  return plan;
}

async function requestRefinementStrategy(text, settings, payload, analysis, baseSolverConfig = {}) {
  const fallbackStrategy = buildEliteGuidedFallbackStrategy(analysis, { ...baseSolverConfig, ...payload?.config }, settings);
  if (!settings.openaiApiKey) {
    return fallbackStrategy;
  }

  try {
    const hotspotSummary = summarizeRefinementHotspots(analysis);
    const parsedResponse = await postOpenAIResponse(settings, {
      model: settings.openaiModel,
      store: false,
      instructions: refinementStrategyPrompt,
      input: [
        `User request: ${text || '(empty)'}`,
        `Problem type: ${payload.problem_type}`,
        `DRL samples: ${payload.config?.drl_samples ?? baseSolverConfig.drl_samples ?? settings.drlSamples}`,
        `Current objective: ${JSON.stringify(baseSolverConfig.objective ?? payload.config?.objective ?? null)}`,
        `Elite-guided hotspot summary: ${JSON.stringify(hotspotSummary)}`,
        `Current solution analysis: ${JSON.stringify(analysis)}`,
      ]
        .filter(Boolean)
        .join('\n'),
    });

    const structuredText = extractStructuredText(getResponseText(parsedResponse));
    if (!structuredText) {
      return fallbackStrategy;
    }
    const parsed = parseJsonObject(structuredText);
    return {
      ...fallbackStrategy,
      ...normalizeRefinementStrategy(parsed.refinement_strategy ?? parsed),
    };
  } catch {
    return fallbackStrategy;
  }
}

async function requestAgentDecision(text, settings, conversation, ingestResult, previousResponseId = null) {
  if (!settings.openaiApiKey) {
    return {
      action: 'reply',
      message: '当前未配置可用的 OpenAI API key，无法让模型判断意图。请先在设置中保存正确的 API key。',
      agentPreviousResponseId: null,
    };
  }

  const parsedResponse = await postOpenAIResponse(settings, {
    model: settings.openaiModel,
    store: false,
    instructions: agentDecisionPrompt,
    input: buildAgentUserPrompt(text, conversation, ingestResult),
  });
  const structuredText = extractStructuredText(getResponseText(parsedResponse));
  if (!structuredText) {
    throw new Error('模型没有返回 agent 决策结果。');
  }

  const decision = parseJsonObject(structuredText);
  return {
    ...decision,
    structuredText,
    agentPreviousResponseId: null,
  };
}

async function requestConversationTitle(settings, context) {
  const fallbackTitle = buildFallbackConversationTitle(context);
  if (!settings.openaiApiKey) {
    return fallbackTitle;
  }

  try {
    const parsedResponse = await postOpenAIResponse(settings, {
      model: settings.openaiModel,
      store: false,
      instructions: conversationTitlePrompt,
      input: [
        `User message: ${context.text || '(empty)'}`,
        Array.isArray(context.uploadedFiles) && context.uploadedFiles.length > 0
          ? `Uploaded files: ${context.uploadedFiles.map((file) => file.name).join(', ')}`
          : null,
        Array.isArray(context.ingestResults) && context.ingestResults.length > 0
          ? `Instance summaries: ${JSON.stringify(context.ingestResults.map((item) => item.summary))}`
          : null,
        context.replyMessage ? `Assistant reply: ${context.replyMessage}` : null,
        context.payload?.problem_type ? `Problem type: ${context.payload.problem_type}` : null,
        context.solveResult ? `Solved distance: ${context.solveResult.final_solution?.distance ?? ''}` : null,
      ]
        .filter(Boolean)
        .join('\n'),
    });

    const structuredText = extractStructuredText(getResponseText(parsedResponse));
    if (!structuredText) {
      return fallbackTitle;
    }
    const parsed = parseJsonObject(structuredText);
    return sanitizeSuggestedTitle(parsed.title, fallbackTitle);
  } catch {
    return fallbackTitle;
  }
}

async function solveRequest(event, request) {
  const startedAt = Date.now();
  const requestId = typeof request?.requestId === 'string' && request.requestId.trim() ? request.requestId.trim() : `req-${Date.now()}`;
  const text = typeof request?.text === 'string' ? request.text.trim() : '';
  const mode = request?.mode === 'thinking' ? 'thinking' : 'quick';
  const settings = resolveEffectiveSettings(request?.settings);
  const conversation = Array.isArray(request?.conversation) ? request.conversation : [];
  const agentPreviousResponseId =
    typeof request?.agentPreviousResponseId === 'string' && request.agentPreviousResponseId.trim()
      ? request.agentPreviousResponseId.trim()
      : null;
  const sender = event.sender;
  const uploadedFiles = normalizeUploadedFiles(request);
  const preparsedIngestResults = Array.isArray(request?.preparsedIngestResults) ? request.preparsedIngestResults : [];

  if (!text && uploadedFiles.length === 0) {
    throw new Error('请输入消息，或先上传实例文件。');
  }

  const ingestResults = [];
  const ingestResultByPath = new Map();
  let payload = null;
  let payloadSource = 'json';
  let structuredText = null;
  let suggestedTitle = null;
  let batchEntries = null;

  emitProgress(sender, requestId, 'receive', '接收请求', 'completed', null);

  for (const entry of preparsedIngestResults) {
    if (!entry || typeof entry !== 'object' || typeof entry.path !== 'string' || !entry.ingestResult || typeof entry.ingestResult !== 'object') {
      continue;
    }
    const normalizedPath = entry.path.trim();
    if (!normalizedPath || ingestResultByPath.has(normalizedPath)) {
      continue;
    }
    ingestResultByPath.set(normalizedPath, entry.ingestResult);
    ingestResults.push(entry.ingestResult);
  }

  if (uploadedFiles.length > 0) {
    for (let index = 0; index < uploadedFiles.length; index += 1) {
      const file = uploadedFiles[index];
      if (ingestResultByPath.has(file.path)) {
        continue;
      }
      emitProgress(
        sender,
        requestId,
        'ingest',
        uploadedFiles.length > 1 ? `识别上传实例 ${index + 1}/${uploadedFiles.length}` : '识别上传实例',
        'running',
        file.name,
      );
      const ingestResult = await runInstanceIngest(file.path);
      ingestResultByPath.set(file.path, ingestResult);
      ingestResults.push(ingestResult);
    }
    emitProgress(
      sender,
      requestId,
      'ingest',
      '识别上传实例',
      'completed',
      uploadedFiles.length > 1 ? `${uploadedFiles.length} 个文件` : ingestResults[0]?.detected_format ?? null,
    );
  }
  emitProgress(sender, requestId, 'decision', '由模型判断是否调用求解 skill', 'running', null);
  const decision = await requestAgentDecision(text, settings, conversation, ingestResults, agentPreviousResponseId);
  const decisionDetail = typeof decision.structuredText === 'string' && decision.structuredText.trim() ? '已由模型返回决策。' : null;

  if (decision.action === 'reply') {
    if (conversation.length === 0) {
      suggestedTitle = await requestConversationTitle(settings, {
        text,
        uploadedFiles,
        ingestResults,
        replyMessage: decision.message,
        payload: null,
        solveResult: null,
      });
    }
    emitProgress(sender, requestId, 'decision', '模型决定不调用求解 skill', 'completed', null);
    emitProgress(sender, requestId, 'reply', '生成普通回复', 'completed', null);
    return {
      kind: 'reply',
      message: typeof decision.message === 'string' && decision.message.trim() ? decision.message.trim() : '我先不调用求解器。请继续说明你的需求。',
      durationMs: Date.now() - startedAt,
      suggestedTitle,
      agentPreviousResponseId: decision.agentPreviousResponseId ?? agentPreviousResponseId,
    };
  }

  emitProgress(sender, requestId, 'decision', '模型决定调用求解 skill', 'completed', decisionDetail);
  structuredText = typeof decision.structuredText === 'string' ? decision.structuredText : null;

  if (decision.use_uploaded_payload && ingestResults.length > 1) {
    batchEntries = ingestResults.map((ingestResult, index) => ({
      fileName: uploadedFiles[index]?.name ?? path.basename(ingestResult.source_path),
      ingestResult,
    }));
    payloadSource = 'upload';
  } else if (decision.use_uploaded_payload && ingestResults.length === 1) {
    payload = ingestResults[0].payload;
    payloadSource = 'upload';
  } else if (decision.payload) {
    payload = decision.payload;
    payloadSource = 'llm';
  } else if (isJsonLike(text)) {
    payload = parseJsonPayload(text);
    payloadSource = 'json';
  } else if (ingestResults.length > 1) {
    batchEntries = ingestResults.map((ingestResult, index) => ({
      fileName: uploadedFiles[index]?.name ?? path.basename(ingestResult.source_path),
      ingestResult,
    }));
    payloadSource = 'upload';
  } else if (ingestResults.length === 1) {
    payload = ingestResults[0].payload;
    payloadSource = 'upload';
  } else {
    throw new Error('模型决定调用求解器，但没有返回有效 payload。');
  }

  const quickMode = mode === 'quick';
  const modelSolverConfig = normalizeSolverConfig(decision.solver_config, quickMode);
  const solverConfig =
    Object.keys(modelSolverConfig).length > 0
      ? modelSolverConfig
      : await requestSolverStrategy(text, settings, mode, payload ?? batchEntries?.[0]?.ingestResult?.payload ?? null, ingestResults);

  if (batchEntries) {
    const { batchItems } = await runBatchSolversWithProgress(
      sender,
      requestId,
      batchEntries,
      mode,
      settings,
      structuredText,
      solverConfig,
    );
    if (conversation.length === 0) {
      suggestedTitle = await requestConversationTitle(settings, {
        text,
        uploadedFiles,
        ingestResults,
        replyMessage: null,
        payload: batchItems[0]?.payload ?? null,
        solveResult: batchItems[0]?.result ?? null,
      });
    }

    const firstBatchItem = batchItems[0];
    return {
      kind: 'solve',
      durationMs: Date.now() - startedAt,
      suggestedTitle,
      agentPreviousResponseId: decision.agentPreviousResponseId ?? agentPreviousResponseId,
      solveResponse: {
        payload: firstBatchItem.payload,
        payloadSource: firstBatchItem.payloadSource,
        result: firstBatchItem.result,
        durationMs: Date.now() - startedAt,
        structuredText,
        ingestResult: firstBatchItem.ingestResult,
        runtimeDefaults: buildRuntimeDefaultsPayload(),
        batchItems,
      },
    };
  }

  let finalPayload = withSolveDefaults(payload, mode, settings, solverConfig);
  let finalResult = null;
  let finalDurationMs = 0;

  if (mode === 'thinking' && settings.openaiApiKey) {
    const warmupSolverConfig = {
      ...solverConfig,
      enable_lookahead: true,
      enable_local_search: false,
      enable_destroy_repair: false,
      enable_vehicle_reduction: false,
      decode_lookahead_as_initial: true,
      tool_plan: buildThinkingWarmupToolPlan(),
    };
    const warmupPayload = withSolveDefaults(payload, mode, settings, warmupSolverConfig);
    emitProgress(sender, requestId, 'solve', '运行 lookahead 预热阶段', 'running', warmupPayload.problem_type);
    const warmupSolve = await runSolverWithProgress(sender, requestId, warmupPayload);
    emitProgress(sender, requestId, 'solve', 'lookahead 预热阶段完成', 'completed', warmupPayload.problem_type);

    const warmupResult = warmupSolve.result;
    const lookaheadBaseSolution = warmupResult.lookahead_solution ?? warmupResult.final_solution;

    emitProgress(sender, requestId, 'analysis', '分析当前解', 'running', '生成路线统计与等待热点');
    const analysis = await runSolutionAnalysis(
      warmupPayload.problem_type,
      warmupPayload.instance,
      lookaheadBaseSolution,
      warmupPayload.config?.objective ?? null,
    );
    emitProgress(sender, requestId, 'analysis', '分析当前解', 'completed', '已生成路线诊断');

    emitProgress(sender, requestId, 'refine_plan', '由模型选择改进算子', 'running', null);
    const requestedRefinement = await requestRefinementStrategy(text, settings, warmupPayload, analysis, solverConfig);
    const refinementStrategy = {
      local_search_operators:
        requestedRefinement.local_search_operators ?? solverConfig.local_search_operators ?? ['or_opt', 'two_opt_star', 'cross_exchange', 'relocate', 'swap', 'two_opt'],
      destroy_repair_operators:
        requestedRefinement.destroy_repair_operators ?? solverConfig.destroy_repair_operators ?? ['shaw_regret', 'route_elimination', 'or_opt', 'relocate', 'swap', 'two_opt'],
      local_search_rounds:
        requestedRefinement.local_search_rounds ??
        solverConfig.local_search_rounds ??
        warmupPayload.config?.local_search_rounds ??
        settings.thinkingLocalSearchRounds,
      destroy_repair_rounds:
        requestedRefinement.destroy_repair_rounds ??
        solverConfig.destroy_repair_rounds ??
        warmupPayload.config?.destroy_repair_rounds ??
        settings.thinkingLocalSearchRounds,
      vehicle_reduction_rounds: requestedRefinement.vehicle_reduction_rounds ?? solverConfig.vehicle_reduction_rounds ?? 10,
      improvement_cycles: requestedRefinement.improvement_cycles ?? 2,
      enable_destroy_repair: requestedRefinement.enable_destroy_repair ?? true,
      enable_vehicle_reduction: requestedRefinement.enable_vehicle_reduction ?? false,
      enable_elite_guided_repair: requestedRefinement.enable_elite_guided_repair ?? solverConfig.enable_elite_guided_repair ?? false,
      allow_worse_acceptance: requestedRefinement.allow_worse_acceptance ?? solverConfig.allow_worse_acceptance ?? true,
      acceptance_budget: requestedRefinement.acceptance_budget ?? solverConfig.acceptance_budget ?? 6,
      acceptance_temperature: requestedRefinement.acceptance_temperature ?? solverConfig.acceptance_temperature ?? 0.01,
      acceptance_decay: requestedRefinement.acceptance_decay ?? solverConfig.acceptance_decay ?? 0.9,
      granular_neighbor_k: requestedRefinement.granular_neighbor_k ?? solverConfig.granular_neighbor_k ?? 24,
      regret_k: requestedRefinement.regret_k ?? solverConfig.regret_k ?? 3,
      shaw_remove_count: requestedRefinement.shaw_remove_count ?? solverConfig.shaw_remove_count ?? 8,
      elite_guided_pool_size: requestedRefinement.elite_guided_pool_size ?? solverConfig.elite_guided_pool_size ?? 24,
      elite_guided_polish_rounds: requestedRefinement.elite_guided_polish_rounds ?? solverConfig.elite_guided_polish_rounds ?? 12,
      elite_guided_seed_trials: requestedRefinement.elite_guided_seed_trials ?? solverConfig.elite_guided_seed_trials ?? 1,
      elite_guided_candidate_count:
        requestedRefinement.elite_guided_candidate_count ?? solverConfig.elite_guided_candidate_count ?? settings.drlSamples,
      objective: requestedRefinement.objective ?? solverConfig.objective ?? warmupPayload.config?.objective ?? null,
    };
    emitProgress(
      sender,
      requestId,
      'refine_plan',
      '由模型选择改进算子',
      'completed',
      `cycles=${refinementStrategy.improvement_cycles}`,
    );

    const refinementPayload = withSolveDefaults(
      {
        ...payload,
        starting_solution: lookaheadBaseSolution,
      },
      mode,
      settings,
      {
        ...solverConfig,
        ...refinementStrategy,
        enable_lookahead: false,
        enable_local_search: true,
        tool_plan: buildRefinementToolPlan(refinementStrategy),
      },
    );
    emitProgress(sender, requestId, 'solve', '运行多轮改进阶段', 'running', refinementPayload.problem_type);
    const refinementSolve = await runSolverWithProgress(sender, requestId, refinementPayload);
    emitProgress(sender, requestId, 'solve', '多轮改进阶段完成', 'completed', refinementPayload.problem_type);

    finalPayload = {
      ...warmupPayload,
      config: refinementPayload.config,
    };
    finalResult = {
      ...warmupResult,
      local_search_solution: refinementSolve.result.local_search_solution ?? refinementSolve.result.final_solution,
      final_solution: refinementSolve.result.final_solution,
      meta: {
        ...warmupResult.meta,
        ...refinementSolve.result.meta,
        lookahead_solution_used_for_refinement: lookaheadBaseSolution,
        warmup_analysis: analysis,
        refinement_strategy: refinementStrategy,
        tool_plan: [
          ...(Array.isArray(warmupResult.meta?.tool_plan) ? warmupResult.meta.tool_plan : []),
          ...(Array.isArray(refinementSolve.result.meta?.tool_plan) ? refinementSolve.result.meta.tool_plan : []),
        ],
        tool_trace: [
          ...(Array.isArray(warmupResult.meta?.tool_trace) ? warmupResult.meta.tool_trace : []),
          ...(Array.isArray(refinementSolve.result.meta?.tool_trace) ? refinementSolve.result.meta.tool_trace : []),
        ],
      },
    };
    finalDurationMs = warmupSolve.durationMs + refinementSolve.durationMs;
  } else {
    emitProgress(sender, requestId, 'solve', '调用本地求解链', 'running', finalPayload.problem_type);
    const solveExecution = await runSolverWithProgress(sender, requestId, finalPayload);
    emitProgress(sender, requestId, 'solve', '本地求解链完成', 'completed', finalPayload.problem_type);
    finalResult = solveExecution.result;
    finalDurationMs = solveExecution.durationMs;
  }

  if (conversation.length === 0) {
    suggestedTitle = await requestConversationTitle(settings, {
      text,
      uploadedFiles,
      ingestResults,
      replyMessage: null,
      payload: finalPayload,
      solveResult: finalResult,
    });
  }

  return {
    kind: 'solve',
    durationMs: Date.now() - startedAt,
    suggestedTitle,
    agentPreviousResponseId: decision.agentPreviousResponseId ?? agentPreviousResponseId,
    solveResponse: {
      payload: finalPayload,
      payloadSource,
      result: finalResult,
      durationMs: finalDurationMs,
      structuredText,
      ingestResult: ingestResults[0] ?? null,
      runtimeDefaults: buildRuntimeDefaultsPayload(),
      batchItems: null,
    },
  };
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 760,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#ffffff',
    title: 'Route AI',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.once('ready-to-show', () => {
    win.show();
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url).catch(() => {});
    return { action: 'deny' };
  });

  win.webContents.on('will-navigate', (event, url) => {
    const currentUrl = win.webContents.getURL();
    if (url === currentUrl || (isDev && url.startsWith(startUrl))) {
      return;
    }
    event.preventDefault();
    shell.openExternal(url).catch(() => {});
  });

  if (isDev) {
    win.loadURL(startUrl);
    return;
  }

  win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
}

app.whenReady().then(() => {
  ipcMain.handle('settings:get', () => loadSettings());
  ipcMain.handle('settings:set', (_event, settings) => saveSettings(settings));
  ipcMain.handle('files:open-instance', async () => {
    const result = await dialog.showOpenDialog({
      title: '导入实例文件',
      properties: ['openFile'],
      filters: [
        {
          name: 'Routing Instances',
          extensions: ['json', 'txt', 'vrp', 'tsp', 'csv', 'tsv'],
        },
      ],
    });
    if (result.canceled || result.filePaths.length === 0) {
      return null;
    }

    const filePath = result.filePaths[0];
    return {
      path: filePath,
      name: path.basename(filePath),
      content: fs.readFileSync(filePath, 'utf8'),
    };
  });
  ipcMain.handle('files:open-instances', async () => {
    const result = await dialog.showOpenDialog({
      title: '导入实例文件',
      properties: ['openFile', 'multiSelections'],
      filters: [
        {
          name: 'Routing Instances',
          extensions: ['json', 'txt', 'vrp', 'tsp', 'csv', 'tsv'],
        },
      ],
    });
    if (result.canceled || result.filePaths.length === 0) {
      return [];
    }

    return result.filePaths.map((filePath) => ({
      path: filePath,
      name: path.basename(filePath),
      content: fs.readFileSync(filePath, 'utf8'),
    }));
  });
  ipcMain.handle('files:save-json', async (_event, request) => {
    const defaultFileName = sanitizeFileName(request?.defaultFileName, 'solve-result.json');
    const result = await dialog.showSaveDialog({
      title: '导出 JSON',
      defaultPath: path.join(app.getPath('downloads'), defaultFileName),
      filters: [{ name: 'JSON Files', extensions: ['json'] }],
    });
    if (result.canceled || !result.filePath) {
      return null;
    }
    fs.writeFileSync(result.filePath, `${JSON.stringify(request?.data ?? {}, null, 2)}\n`, 'utf8');
    return result.filePath;
  });
  ipcMain.handle('solver:ingest-file', async (_event, request) => {
    if (!request || typeof request.path !== 'string' || !request.path.trim()) {
      throw new Error('Missing ingest file path.');
    }
    return runCancelableInstanceIngest(request.requestId, request.path.trim());
  });
  ipcMain.handle('solver:cancel-ingest', (_event, requestId) => cancelInstanceIngest(requestId));
  ipcMain.handle('solver:solve', (event, request) => solveRequest(event, request));

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
