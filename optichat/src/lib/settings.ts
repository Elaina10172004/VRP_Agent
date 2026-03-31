import type { AppSettings } from '../types/settings';

export const defaultSettings: AppSettings = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  hasStoredOpenaiApiKey: false,
  openaiApiKeyLast4: '',
  openaiModel: 'gpt-5.4',
  drlSamples: 128,
  enableLookahead: true,
  lookaheadTopK: 3,
  lookaheadConfidentProb: 0.95,
  enableLocalSearch: true,
  fastLocalSearchRounds: 8,
  thinkingLocalSearchRounds: 50,
};

const SETTINGS_STORAGE_KEY = 'optichat:settings';

function sanitizeNumber(value: number, fallback: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

function sanitizeFloat(value: number, fallback: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, value));
}

export function sanitizeSettings(input: Partial<AppSettings> | null | undefined): AppSettings {
  const source = input ?? {};
  const legacyLocalSearchRounds = source.localSearchRounds;
  const legacyLookaheadTopK = source.lookaheadTopK ?? source.lookaheadDepth;
  const legacyLookaheadConfidentProb = source.lookaheadConfidentProb;
  const requestedModel =
    typeof source.openaiModel === 'string' && source.openaiModel.trim()
      ? source.openaiModel.trim()
      : defaultSettings.openaiModel;
  return {
    openaiBaseUrl:
      typeof source.openaiBaseUrl === 'string' && source.openaiBaseUrl.trim()
        ? source.openaiBaseUrl.trim()
        : defaultSettings.openaiBaseUrl,
    openaiApiKey: typeof source.openaiApiKey === 'string' ? source.openaiApiKey.trim() : defaultSettings.openaiApiKey,
    hasStoredOpenaiApiKey:
      typeof source.hasStoredOpenaiApiKey === 'boolean' ? source.hasStoredOpenaiApiKey : defaultSettings.hasStoredOpenaiApiKey,
    openaiApiKeyLast4:
      typeof source.openaiApiKeyLast4 === 'string' ? source.openaiApiKeyLast4.trim().slice(-4) : defaultSettings.openaiApiKeyLast4,
    openaiModel: requestedModel === 'gpt-5.4-mini' ? 'gpt-5.4' : requestedModel,
    drlSamples: sanitizeNumber(source.drlSamples ?? defaultSettings.drlSamples, defaultSettings.drlSamples, 1, 2048),
    enableLookahead: source.enableLookahead ?? defaultSettings.enableLookahead,
    lookaheadTopK: sanitizeNumber(
      legacyLookaheadTopK ?? defaultSettings.lookaheadTopK,
      defaultSettings.lookaheadTopK,
      1,
      32,
    ),
    lookaheadConfidentProb: sanitizeFloat(
      legacyLookaheadConfidentProb ?? defaultSettings.lookaheadConfidentProb,
      defaultSettings.lookaheadConfidentProb,
      0.5,
      0.999,
    ),
    enableLocalSearch: source.enableLocalSearch ?? defaultSettings.enableLocalSearch,
    fastLocalSearchRounds: sanitizeNumber(
      source.fastLocalSearchRounds ?? legacyLocalSearchRounds ?? defaultSettings.fastLocalSearchRounds,
      defaultSettings.fastLocalSearchRounds,
      1,
      500,
    ),
    thinkingLocalSearchRounds: sanitizeNumber(
      source.thinkingLocalSearchRounds ?? legacyLocalSearchRounds ?? defaultSettings.thinkingLocalSearchRounds,
      defaultSettings.thinkingLocalSearchRounds,
      1,
      500,
    ),
  };
}

export async function loadSettings(): Promise<AppSettings> {
  if (window.desktopApp?.settings?.get) {
    return sanitizeSettings(await window.desktopApp.settings.get());
  }

  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    return sanitizeSettings(raw ? JSON.parse(raw) : undefined);
  } catch {
    return { ...defaultSettings };
  }
}

export async function saveSettings(settings: AppSettings): Promise<AppSettings> {
  const normalized = sanitizeSettings(settings);

  if (window.desktopApp?.settings?.save) {
    return sanitizeSettings(await window.desktopApp.settings.save(normalized));
  }

  window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}
