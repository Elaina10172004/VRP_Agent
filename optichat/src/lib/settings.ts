import type { AppSettings } from '../types/settings';

export const defaultSettings: AppSettings = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  openaiModel: 'gpt-5.4-mini',
  drlSamples: 128,
  enableLookahead: true,
  lookaheadDepth: 2,
  lookaheadBeamWidth: 4,
  enableLocalSearch: false,
  localSearchRounds: 50,
};

const SETTINGS_STORAGE_KEY = 'optichat:settings';

function sanitizeNumber(value: number, fallback: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

export function sanitizeSettings(input: Partial<AppSettings> | null | undefined): AppSettings {
  const source = input ?? {};
  return {
    openaiBaseUrl:
      typeof source.openaiBaseUrl === 'string' && source.openaiBaseUrl.trim()
        ? source.openaiBaseUrl.trim()
        : defaultSettings.openaiBaseUrl,
    openaiApiKey: typeof source.openaiApiKey === 'string' ? source.openaiApiKey.trim() : defaultSettings.openaiApiKey,
    openaiModel:
      typeof source.openaiModel === 'string' && source.openaiModel.trim()
        ? source.openaiModel.trim()
        : defaultSettings.openaiModel,
    drlSamples: sanitizeNumber(source.drlSamples ?? defaultSettings.drlSamples, defaultSettings.drlSamples, 1, 2048),
    enableLookahead: source.enableLookahead ?? defaultSettings.enableLookahead,
    lookaheadDepth: sanitizeNumber(source.lookaheadDepth ?? defaultSettings.lookaheadDepth, defaultSettings.lookaheadDepth, 1, 5),
    lookaheadBeamWidth: sanitizeNumber(
      source.lookaheadBeamWidth ?? defaultSettings.lookaheadBeamWidth,
      defaultSettings.lookaheadBeamWidth,
      1,
      64,
    ),
    enableLocalSearch: source.enableLocalSearch ?? defaultSettings.enableLocalSearch,
    localSearchRounds: sanitizeNumber(
      source.localSearchRounds ?? defaultSettings.localSearchRounds,
      defaultSettings.localSearchRounds,
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
