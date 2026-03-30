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

const defaultSettings = Object.freeze({
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  openaiModel: 'gpt-5.4-mini',
  drlSamples: 128,
  enableLookahead: true,
  lookaheadDepth: 2,
  lookaheadBeamWidth: 4,
  enableLocalSearch: false,
  localSearchRounds: 50,
});

const ALLOWED_TOOL_PLAN_STEPS = new Set([
  'construct_initial',
  'validate_solution',
  'reduce_vehicles',
  'apply_lookahead',
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
]);

const agentDecisionPrompt = [
  'You are the orchestration agent for a desktop routing assistant.',
  'Decide whether the local solve skill should be called, and optionally attach a solver strategy.',
  'Return JSON only. No markdown, no code fences, no commentary.',
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
  'Allowed tool_plan steps: construct_initial, validate_solution, reduce_vehicles, apply_lookahead, destroy_repair, improve_solution, compare_solutions.',
  'Allowed operators: two_opt, relocate, swap, or_opt, two_opt_star, cross_exchange, route_elimination.',
  'You may also set solver_config.objective, for example {"primary":"vehicle_count","vehicle_fixed_cost":100.0,"lateness_penalty":10.0}.',
  'Every plan must start with construct_initial and end with compare_solutions.',
  'Insert validate_solution after each major action.',
  'For TSP never use reduce_vehicles or route_elimination.',
  'For CVRP and CVRPTW you may use reduce_vehicles and route_elimination.',
  'Prefer compact plans. Only enable expensive steps when they are likely to help.',
  'If the user explicitly wants speed, keep the plan short.',
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
  return {
    openaiBaseUrl:
      typeof source.openaiBaseUrl === 'string' && source.openaiBaseUrl.trim()
        ? source.openaiBaseUrl.trim()
        : defaultSettings.openaiBaseUrl,
    openaiApiKey: typeof source.openaiApiKey === 'string' ? source.openaiApiKey.trim().slice(0, 4096) : '',
    openaiModel:
      typeof source.openaiModel === 'string' && source.openaiModel.trim()
        ? source.openaiModel.trim().slice(0, 256)
        : defaultSettings.openaiModel,
    drlSamples: clampInt(source.drlSamples, 1, 2048, defaultSettings.drlSamples),
    enableLookahead: typeof source.enableLookahead === 'boolean' ? source.enableLookahead : defaultSettings.enableLookahead,
    lookaheadDepth: clampInt(source.lookaheadDepth, 1, 8, defaultSettings.lookaheadDepth),
    lookaheadBeamWidth: clampInt(source.lookaheadBeamWidth, 1, 64, defaultSettings.lookaheadBeamWidth),
    enableLocalSearch: typeof source.enableLocalSearch === 'boolean' ? source.enableLocalSearch : defaultSettings.enableLocalSearch,
    localSearchRounds: clampInt(source.localSearchRounds, 1, 1000, defaultSettings.localSearchRounds),
  };
}

function loadSettings() {
  const filePath = resolveSettingsFile();
  if (!fs.existsSync(filePath)) {
    return { ...defaultSettings };
  }

  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const normalized = sanitizeSettings(raw);
    if (raw.openaiApiKeySecure && safeStorage && safeStorage.isEncryptionAvailable()) {
      try {
        normalized.openaiApiKey = safeStorage.decryptString(Buffer.from(raw.openaiApiKeySecure, 'base64'));
      } catch {
        normalized.openaiApiKey = '';
      }
    } else if (typeof raw.openaiApiKey === 'string') {
      normalized.openaiApiKey = raw.openaiApiKey;
    }
    return normalized;
  } catch {
    return { ...defaultSettings };
  }
}

function saveSettings(input) {
  const normalized = sanitizeSettings(input);
  const filePath = resolveSettingsFile();
  const persisted = { ...normalized };

  if (safeStorage && safeStorage.isEncryptionAvailable() && normalized.openaiApiKey) {
    const encrypted = safeStorage.encryptString(normalized.openaiApiKey);
    persisted.openaiApiKeySecure = encrypted.toString('base64');
    delete persisted.openaiApiKey;
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(persisted, null, 2), 'utf8');
  return normalized;
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

function resolveChatCompletionsUrl(baseUrl) {
  const trimmed = String(baseUrl || '').trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'https://api.openai.com/v1/chat/completions';
  }
  return trimmed.endsWith('/chat/completions') ? trimmed : `${trimmed}/chat/completions`;
}

function getMessageText(choice) {
  const content = choice?.message?.content;
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') {
          return part;
        }
        if (part && typeof part.text === 'string') {
          return part.text;
        }
        return '';
      })
      .join('');
  }
  return '';
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
  const snippet = String(responseText || '').replace(/\s+/g, ' ').slice(0, 160);
  return new Error(
    `OpenAI 接口返回了 HTML 页面，而不是 JSON。通常是 Base URL 配错了，或代理/登录页拦截了请求。\n请求地址：${url}\nContent-Type：${
      contentType || 'unknown'
    }\n响应片段：${snippet}`,
  );
}

function parseJsonHttpResponse(url, response, responseText) {
  const contentType = response.headers.get('content-type') || '';
  const trimmed = String(responseText || '').trim();
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
    throw new Error(`OpenAI 请求失败：${response.status} ${response.statusText}\n${String(responseText || '').slice(0, 600)}`);
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

async function postOpenAIChat(settings, body) {
  const url = resolveChatCompletionsUrl(settings.openaiBaseUrl);
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${settings.openaiApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  const responseText = await response.text();
  return parseJsonHttpResponse(url, response, responseText);
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
    return ['construct_initial', 'validate_solution', 'compare_solutions'];
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
  }

  const integerFields = [
    ['drl_samples', 1, 2048],
    ['seed_trials', 1, 32],
    ['initial_candidate_count', 1, 128],
    ['lookahead_depth', 1, 8],
    ['lookahead_beam_width', 1, 64],
    ['lookahead_per_operator_limit', 1, 64],
    ['local_search_rounds', 1, 1000],
    ['destroy_repair_rounds', 1, 1000],
    ['vehicle_reduction_rounds', 1, 1000],
  ];

  for (const [field, min, max] of integerFields) {
    if (value[field] !== undefined && value[field] !== null && value[field] !== '') {
      normalized[field] = clampInt(value[field], min, max, min);
    }
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

  return {
    ...payload,
    problem_type: String(payload.problem_type).trim().toLowerCase(),
    config: {
      ...rawConfig,
      mode: quickMode ? 'fast' : 'hybrid',
      drl_samples: clampInt(rawConfig.drl_samples, 1, 2048, settings.drlSamples),
      seed_trials: clampInt(rawConfig.seed_trials, 1, 32, defaultSeedTrials),
      enable_lookahead: quickMode ? false : rawConfig.enable_lookahead ?? settings.enableLookahead,
      lookahead_depth: clampInt(rawConfig.lookahead_depth, 1, 8, settings.lookaheadDepth),
      lookahead_beam_width: clampInt(rawConfig.lookahead_beam_width, 1, 64, settings.lookaheadBeamWidth),
      lookahead_k: clampInt(rawConfig.lookahead_k, 1, 64, 1),
      enable_local_search: quickMode ? false : rawConfig.enable_local_search ?? settings.enableLocalSearch,
      local_search_rounds: clampInt(rawConfig.local_search_rounds, 1, 1000, settings.localSearchRounds),
      ...(rawConfig.tool_plan ? { tool_plan: rawConfig.tool_plan } : {}),
      ...(rawConfig.lookahead_operators ? { lookahead_operators: rawConfig.lookahead_operators } : {}),
      ...(rawConfig.local_search_operators ? { local_search_operators: rawConfig.local_search_operators } : {}),
      ...(rawConfig.destroy_repair_operators ? { destroy_repair_operators: rawConfig.destroy_repair_operators } : {}),
      ...(rawConfig.objective ? { objective: rawConfig.objective } : {}),
      ...(rawConfig.enable_vehicle_reduction !== undefined ? { enable_vehicle_reduction: Boolean(rawConfig.enable_vehicle_reduction) } : {}),
      ...(rawConfig.enable_destroy_repair !== undefined ? { enable_destroy_repair: Boolean(rawConfig.enable_destroy_repair) } : {}),
      ...(rawConfig.initial_candidate_count !== undefined
        ? { initial_candidate_count: clampInt(rawConfig.initial_candidate_count, 1, 128, 1) }
        : {}),
      ...(rawConfig.lookahead_per_operator_limit !== undefined
        ? { lookahead_per_operator_limit: clampInt(rawConfig.lookahead_per_operator_limit, 1, 64, settings.lookaheadBeamWidth) }
        : {}),
      ...(rawConfig.destroy_repair_rounds !== undefined
        ? { destroy_repair_rounds: clampInt(rawConfig.destroy_repair_rounds, 1, 1000, settings.localSearchRounds) }
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

async function runInstanceIngest(filePath) {
  const execution = await runJsonModule('instance_skill.cli', {
    args: ['--input-file', filePath],
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

async function requestSolverStrategy(text, settings, mode, payload, ingestResults) {
  if (!settings.openaiApiKey || mode !== 'thinking') {
    return {};
  }

  try {
    const parsedResponse = await postOpenAIChat(settings, {
      model: settings.openaiModel,
      temperature: 0.1,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: solverStrategyPrompt },
        {
          role: 'user',
          content: [
            `User request: ${text || '(empty)'}`,
            `Mode: ${mode}`,
            `Payload summary: ${JSON.stringify(summarizePayloadForStrategy(payload))}`,
            Array.isArray(ingestResults) && ingestResults.length > 0
              ? `Uploaded summaries: ${JSON.stringify(ingestResults.map((item) => item.summary).filter(Boolean))}`
              : null,
          ]
            .filter(Boolean)
            .join('\n'),
        },
      ],
    });

    const structuredText = extractStructuredText(getMessageText(parsedResponse?.choices?.[0]));
    if (!structuredText) {
      return {};
    }

    const parsed = parseJsonObject(structuredText);
    return normalizeSolverConfig(parsed.solver_config ?? parsed, false);
  } catch {
    return {};
  }
}

function hasExplicitSolveIntent(text) {
  const content = String(text || '').trim();
  return /(全部求解|批量求解|直接求解|开始求解|求解上传|求解这些|求解这批|solve all|solve uploaded|solve these|optimi[sz]e uploaded)/i.test(content);
}

function maybeShortCircuitSolveDecision(text, ingestResult) {
  const hasUploads = Array.isArray(ingestResult) ? ingestResult.length > 0 : Boolean(ingestResult);
  if (!hasUploads) {
    return null;
  }

  if (!String(text || '').trim()) {
    return { action: 'solve', use_uploaded_payload: true, shortCircuited: true };
  }

  if (hasExplicitSolveIntent(text)) {
    return { action: 'solve', use_uploaded_payload: true, shortCircuited: true };
  }

  return null;
}

function heuristicAgentDecision(text, ingestResult) {
  const content = String(text || '').trim();
  const solveIntent = /(求解|优化|路线|路径|排程|构造解|跑一下|运行|drl|lookahead|局部搜索|solve|optimi[sz]e|vrp|tsp|cvrp|cvrptw)/i.test(content);
  const explainIntent = /(解释|说明|是什么|为什么|怎么|如何|介绍|必要|有必要|rag|skill|设置)/i.test(content);
  const hasUploads = Array.isArray(ingestResult) ? ingestResult.length > 0 : Boolean(ingestResult);

  if (!content && hasUploads) {
    return {
      action: 'reply',
      message: '我已经收到上传的实例文件。请明确说明是要求解、分析、转换格式，还是解释这个实例。',
    };
  }
  if (hasUploads && solveIntent) {
    return { action: 'solve', use_uploaded_payload: true };
  }
  if (explainIntent && !solveIntent) {
    return {
      action: 'reply',
      message: '当前输入更像普通对话或说明请求，我不会直接调用求解 skill。若你要我求解，请明确说“求解这个实例”或“优化上传实例”。',
    };
  }
  if (solveIntent) {
    return {
      action: 'reply',
      message: '我判断你是想求解，但当前还缺少可直接求解的结构化信息。请上传实例文件、粘贴 JSON payload，或配置 OpenAI Key 让我先做结构化。',
    };
  }
  return {
    action: 'reply',
    message: '我先不调用求解 skill。你可以继续提问，或明确说明需要我对哪个实例执行求解。',
  };
}

async function requestAgentDecision(text, settings, conversation, ingestResult) {
  const shortCircuitDecision = maybeShortCircuitSolveDecision(text, ingestResult);
  if (shortCircuitDecision) {
    return shortCircuitDecision;
  }

  if (!settings.openaiApiKey) {
    return heuristicAgentDecision(text, ingestResult);
  }

  const parsedResponse = await postOpenAIChat(settings, {
    model: settings.openaiModel,
    temperature: 0.1,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: agentDecisionPrompt },
      { role: 'user', content: buildAgentUserPrompt(text, conversation, ingestResult) },
    ],
  });
  const structuredText = extractStructuredText(getMessageText(parsedResponse?.choices?.[0]));
  if (!structuredText) {
    throw new Error('模型没有返回 agent 决策结果。');
  }

  const decision = parseJsonObject(structuredText);
  return {
    ...decision,
    structuredText,
  };
}

async function requestConversationTitle(settings, context) {
  const fallbackTitle = buildFallbackConversationTitle(context);
  if (!settings.openaiApiKey) {
    return fallbackTitle;
  }

  try {
    const parsedResponse = await postOpenAIChat(settings, {
      model: settings.openaiModel,
      temperature: 0.2,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: conversationTitlePrompt },
        {
          role: 'user',
          content: [
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
        },
      ],
    });

    const structuredText = extractStructuredText(getMessageText(parsedResponse?.choices?.[0]));
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
  const settings = sanitizeSettings(request?.settings);
  const conversation = Array.isArray(request?.conversation) ? request.conversation : [];
  const sender = event.sender;
  const uploadedFiles = normalizeUploadedFiles(request);

  if (!text && uploadedFiles.length === 0) {
    throw new Error('请输入消息，或先上传实例文件。');
  }

  const ingestResults = [];
  let payload = null;
  let payloadSource = 'json';
  let structuredText = null;
  let suggestedTitle = null;
  let batchEntries = null;

  emitProgress(sender, requestId, 'receive', '接收请求', 'completed', null);

  if (uploadedFiles.length > 0) {
    for (let index = 0; index < uploadedFiles.length; index += 1) {
      const file = uploadedFiles[index];
      emitProgress(
        sender,
        requestId,
        'ingest',
        uploadedFiles.length > 1 ? `识别上传实例 ${index + 1}/${uploadedFiles.length}` : '识别上传实例',
        'running',
        file.name,
      );
      ingestResults.push(await runInstanceIngest(file.path));
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
  const decision = await requestAgentDecision(text, settings, conversation, ingestResults);
  const decisionDetail = decision.shortCircuited
    ? !text
      ? '检测到已上传实例，直接开始求解。'
      : '检测到明确求解指令，直接使用已上传实例求解。'
    : null;

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

  const normalizedPayload = withSolveDefaults(payload, mode, settings, solverConfig);
  emitProgress(sender, requestId, 'solve', '调用本地求解链', 'running', normalizedPayload.problem_type);
  const solveResult = await runSolverWithProgress(sender, requestId, normalizedPayload);
  emitProgress(sender, requestId, 'solve', '本地求解链完成', 'completed', normalizedPayload.problem_type);

  if (conversation.length === 0) {
    suggestedTitle = await requestConversationTitle(settings, {
      text,
      uploadedFiles,
      ingestResults,
      replyMessage: null,
      payload: normalizedPayload,
      solveResult: solveResult.result,
    });
  }

  return {
    kind: 'solve',
    durationMs: Date.now() - startedAt,
    suggestedTitle,
    solveResponse: {
      payload: normalizedPayload,
      payloadSource,
      result: solveResult.result,
      durationMs: solveResult.durationMs,
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
    title: 'VRP Agent',
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
