export type AppSettings = {
  openaiBaseUrl: string;
  openaiApiKey: string;
  openaiModel: string;
  drlSamples: number;
  enableLookahead: boolean;
  lookaheadDepth: number;
  lookaheadBeamWidth: number;
  enableLocalSearch: boolean;
  localSearchRounds: number;
};
