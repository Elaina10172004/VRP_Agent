import type { AppSettings } from './settings';
import type {
  DesktopAgentResponse,
  DesktopIngestRequest,
  DesktopOpenFileResponse,
  DesktopPreparedIngestResult,
  DesktopProgressEvent,
  DesktopSaveJsonRequest,
  DesktopSolveRequest,
  DesktopIngestResult,
} from './solver';

declare global {
  interface Window {
    desktopApp?: {
      platform: string;
      versions: {
        chrome: string;
        electron: string;
        node: string;
      };
      settings?: {
        get: () => Promise<AppSettings>;
        save: (settings: AppSettings) => Promise<AppSettings>;
      };
      files?: {
        openInstance: () => Promise<DesktopOpenFileResponse | null>;
        openInstances: () => Promise<DesktopOpenFileResponse[]>;
        saveJson: (request: DesktopSaveJsonRequest) => Promise<string | null>;
      };
      solver?: {
        ingestFile: (request: DesktopIngestRequest) => Promise<DesktopIngestResult>;
        cancelIngest: (requestId: string) => Promise<boolean>;
        solve: (request: DesktopSolveRequest) => Promise<DesktopAgentResponse>;
        onProgress: (callback: (event: DesktopProgressEvent) => void) => () => void;
      };
    };
  }
}

export {};
