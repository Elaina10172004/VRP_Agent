export type AppSettings = {
  openaiBaseUrl: string;
  openaiApiKey: string;
  hasStoredOpenaiApiKey?: boolean;
  openaiApiKeyLast4?: string;
  openaiModel: string;
  drlSamples: number;
  enableLookahead: boolean;
  lookaheadTopK: number;
  lookaheadConfidentProb: number;
  enableLocalSearch: boolean;
  fastLocalSearchRounds: number;
  thinkingLocalSearchRounds: number;
  localSearchRounds?: number;
  lookaheadDepth?: number;
  lookaheadBeamWidth?: number;
  lookaheadChunkSize?: number;
};
