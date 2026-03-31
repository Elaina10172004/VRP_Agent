import React, { useEffect, useMemo, useState } from 'react';
import { ChatThread } from './components/ChatThread';
import { Header } from './components/Header';
import { HistorySidebar } from './components/HistorySidebar';
import { InputArea, type InputAttachmentItem } from './components/InputArea';
import { ResultView } from './components/ResultView';
import { SettingsModal } from './components/SettingsModal';
import { SolvingProcess } from './components/SolvingProcess';
import {
  appendAssistantReplyMessage,
  appendAssistantSolveMessage,
  appendUserTurn,
  buildConversationContext,
  createEmptySession,
  createId,
  ensureSessions,
  getLatestSolveResponse,
  loadActiveSessionId,
  loadChatSessions,
  saveActiveSessionId,
  saveChatSessions,
  type AssistantSolveTurn,
} from './lib/history';
import { defaultSettings, loadSettings, saveSettings } from './lib/settings';
import type { AppSettings } from './types/settings';
import type {
  DesktopAgentResponse,
  DesktopIngestResult,
  DesktopPreparedIngestResult,
  DesktopProgressEvent,
  DesktopSolveResponse,
  DesktopUploadedFileRef,
  SolveMode,
} from './types/solver';

type AppState = 'idle' | 'solving' | 'result';

type UploadItem = {
  id: string;
  path: string;
  name: string;
  status: 'parsing' | 'ready' | 'error';
  ingestResult: DesktopIngestResult | null;
  parseRequestId: string | null;
  error: string | null;
};

const initialSessions = ensureSessions(loadChatSessions());
const initialActiveSessionId = (() => {
  const stored = loadActiveSessionId();
  return stored && initialSessions.some((session) => session.id === stored) ? stored : initialSessions[0].id;
})();

function upsertProgressEvent(events: DesktopProgressEvent[], nextEvent: DesktopProgressEvent): DesktopProgressEvent[] {
  const index = events.findIndex((event) => event.stepId === nextEvent.stepId);
  if (index === -1) {
    return [...events, nextEvent];
  }

  const copied = [...events];
  copied[index] = nextEvent;
  return copied;
}

function normalizeSessionTitle(title: string | null | undefined): string | null {
  if (typeof title !== 'string') {
    return null;
  }
  const normalized = title.trim().replace(/\s+/g, ' ').replace(/^["']+|["']+$/g, '');
  if (!normalized) {
    return null;
  }
  return normalized.length > 24 ? normalized.slice(0, 24) : normalized;
}

function isIngestCancelled(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /INGEST_CANCELLED/i.test(message);
}

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [mode, setMode] = useState<SolveMode>('quick');
  const [inputText, setInputText] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<UploadItem[]>([]);
  const [isPickingFiles, setIsPickingFiles] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeResult, setActiveResult] = useState<DesktopSolveResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sessions, setSessions] = useState(initialSessions);
  const [currentSessionId, setCurrentSessionId] = useState(initialActiveSessionId);
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null);
  const [progressEvents, setProgressEvents] = useState<DesktopProgressEvent[]>([]);
  const [pendingAssistantSessionId, setPendingAssistantSessionId] = useState<string | null>(null);

  useEffect(() => {
    loadSettings().then(setSettings).catch(() => setSettings(defaultSettings));
  }, []);

  useEffect(() => {
    saveChatSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveActiveSessionId(currentSessionId);
  }, [currentSessionId]);

  useEffect(() => {
    const unsubscribe = window.desktopApp?.solver?.onProgress?.((event) => {
      setProgressEvents((previous) => {
        if (!activeRequestId || event.requestId !== activeRequestId) {
          return previous;
        }
        if (['solve', 'seed', 'lookahead', 'local_search', 'finalize'].includes(event.stepId)) {
          setAppState('solving');
        }
        return upsertProgressEvent(previous, event);
      });
    });

    return () => {
      unsubscribe?.();
    };
  }, [activeRequestId]);

  const currentSession = useMemo(() => {
    return sessions.find((session) => session.id === currentSessionId) ?? sessions[0];
  }, [currentSessionId, sessions]);

  const latestSessionResult = getLatestSolveResponse(currentSession);

  const attachmentViewModels = useMemo<InputAttachmentItem[]>(
    () =>
      uploadedFiles.map((file) => ({
        id: file.id,
        name: file.name,
        status: file.status,
        error: file.error,
      })),
    [uploadedFiles],
  );

  const handleSaveSettings = async (nextSettings: AppSettings) => {
    const persisted = await saveSettings(nextSettings);
    setSettings(persisted);
  };

  const startBackgroundIngest = (file: DesktopUploadedFileRef, itemId: string, parseRequestId: string) => {
    void window.desktopApp?.solver
      ?.ingestFile({ requestId: parseRequestId, path: file.path })
      .then((ingestResult) => {
        setUploadedFiles((previous) =>
          previous.map((item) =>
            item.id === itemId && item.parseRequestId === parseRequestId
              ? { ...item, status: 'ready', ingestResult, parseRequestId: null, error: null }
              : item,
          ),
        );
      })
      .catch((error) => {
        if (isIngestCancelled(error)) {
          return;
        }
        const message = error instanceof Error ? error.message : '实例解析失败';
        setUploadedFiles((previous) =>
          previous.map((item) =>
            item.id === itemId && item.parseRequestId === parseRequestId
              ? { ...item, status: 'error', ingestResult: null, parseRequestId: null, error: message }
              : item,
          ),
        );
      });
  };

  const handlePickFiles = async () => {
    setIsPickingFiles(true);
    try {
      const files = (await window.desktopApp?.files?.openInstances?.()) ?? [];
      if (files.length === 0) {
        return;
      }

      const existingPaths = new Set(uploadedFiles.map((file) => file.path));
      const freshFiles = files.filter((file) => !existingPaths.has(file.path));
      if (freshFiles.length === 0) {
        return;
      }

      const queued = freshFiles.map((file) => {
        const itemId = createId('upload');
        const parseRequestId = createId('ingest');
        return {
          file,
          item: {
            id: itemId,
            path: file.path,
            name: file.name,
            status: 'parsing' as const,
            ingestResult: null,
            parseRequestId,
            error: null,
          },
        };
      });

      setUploadedFiles((previous) => [...previous, ...queued.map((entry) => entry.item)]);
      setErrorMessage(null);

      for (const entry of queued) {
        startBackgroundIngest(entry.file, entry.item.id, entry.item.parseRequestId);
      }
    } finally {
      setIsPickingFiles(false);
    }
  };

  const handleCreateSession = () => {
    const nextSession = createEmptySession();
    setSessions((previous) => [nextSession, ...previous]);
    setCurrentSessionId(nextSession.id);
    setInputText('');
    setUploadedFiles([]);
    setIsPickingFiles(false);
    setErrorMessage(null);
    setActiveResult(null);
    setActiveRequestId(null);
    setProgressEvents([]);
    setPendingAssistantSessionId(null);
    setAppState('idle');
  };

  const handleSelectSession = (sessionId: string) => {
    setCurrentSessionId(sessionId);
    setInputText('');
    setUploadedFiles([]);
    setIsPickingFiles(false);
    setErrorMessage(null);
    setActiveResult(null);
    setActiveRequestId(null);
    setProgressEvents([]);
    setPendingAssistantSessionId(null);
    setAppState('idle');
  };

  const handleDeleteSession = (sessionId: string) => {
    const filtered = sessions.filter((session) => session.id !== sessionId);
    const nextSessions = filtered.length > 0 ? filtered : [createEmptySession()];
    setSessions(nextSessions);

    if (currentSessionId === sessionId) {
      setCurrentSessionId(nextSessions[0].id);
      setInputText('');
      setUploadedFiles([]);
      setIsPickingFiles(false);
      setErrorMessage(null);
      setActiveResult(null);
      setActiveRequestId(null);
      setProgressEvents([]);
      setPendingAssistantSessionId(null);
      setAppState('idle');
    }
  };

  const handleAgentResponse = (response: DesktopAgentResponse, requestId: string, sessionId: string) => {
    const suggestedTitle = normalizeSessionTitle(response.suggestedTitle);

    if (response.kind === 'reply') {
      setSessions((previous) =>
        previous.map((session) =>
          session.id === sessionId
            ? (() => {
                const appended = appendAssistantReplyMessage(session, response.message, response.agentPreviousResponseId);
                return suggestedTitle && session.title === '鏂板璇?' ? { ...appended, title: suggestedTitle } : appended;
              })()
            : session,
        ),
      );
      setPendingAssistantSessionId((current) => (current === sessionId ? null : current));
      setActiveRequestId(null);
      setAppState('idle');
      return;
    }

    setSessions((previous) =>
      previous.map((session) =>
        session.id === sessionId
          ? (() => {
              const appended = appendAssistantSolveMessage(session, response.solveResponse, response.agentPreviousResponseId);
              return suggestedTitle && session.title === '鏂板璇?' ? { ...appended, title: suggestedTitle } : appended;
            })()
          : session,
      ),
    );

    setPendingAssistantSessionId((current) => (current === sessionId ? null : current));
    setActiveResult(response.solveResponse);
    setActiveRequestId(null);
    setAppState('result');
    setProgressEvents((previous) =>
      upsertProgressEvent(previous, {
        requestId,
        stepId: 'complete',
        label: '完成',
        status: 'completed',
        detail: null,
        timestamp: new Date().toISOString(),
      }),
    );
  };

  const handleRemoveSelectedFile = (index: number) => {
    setUploadedFiles((previous) => {
      const target = previous[index];
      if (target?.status === 'parsing' && target.parseRequestId) {
        void window.desktopApp?.solver?.cancelIngest?.(target.parseRequestId);
      }
      return previous.filter((_, fileIndex) => fileIndex !== index);
    });
  };

  const handleSend = async () => {
    const text = inputText.trim();
    const usableUploads = uploadedFiles.filter((file) => file.status !== 'error');

    if (!text && usableUploads.length === 0) {
      return;
    }

    if (!window.desktopApp?.solver?.solve) {
      setErrorMessage('当前桌面环境没有可用的求解接口，请通过 Electron 启动应用。');
      return;
    }

    const requestId = createId('request');
    const outgoingFiles: DesktopUploadedFileRef[] = usableUploads.map((file) => ({ path: file.path, name: file.name }));
    const preparsedIngestResults: DesktopPreparedIngestResult[] = usableUploads
      .filter((file): file is UploadItem & { ingestResult: DesktopIngestResult } => file.status === 'ready' && Boolean(file.ingestResult))
      .map((file) => ({
        path: file.path,
        ingestResult: file.ingestResult,
      }));
    const parsingUploads = usableUploads.filter((file) => file.status === 'parsing' && file.parseRequestId);
    const sessionWithUser = appendUserTurn(currentSession, {
      text,
      uploadedFiles: outgoingFiles,
      mode,
    });

    setSessions((previous) => previous.map((session) => (session.id === currentSessionId ? sessionWithUser : session)));
    setPendingAssistantSessionId(currentSessionId);
    setErrorMessage(null);
    setActiveResult(null);
    setActiveRequestId(requestId);
    setProgressEvents([]);
    setAppState('idle');
    setInputText('');
    setUploadedFiles([]);

    for (const file of parsingUploads) {
      void window.desktopApp?.solver?.cancelIngest?.(file.parseRequestId as string);
    }

    try {
      const response = await window.desktopApp.solver.solve({
        requestId,
        text,
        mode,
        settings,
        uploadedFiles: outgoingFiles,
        preparsedIngestResults,
        conversation: buildConversationContext(sessionWithUser),
        agentPreviousResponseId: sessionWithUser.agentPreviousResponseId,
      });

      handleAgentResponse(response, requestId, currentSessionId);
    } catch (error) {
      const message = error instanceof Error ? error.message : '调用求解链失败，请检查 OpenAI 配置或本地脚本环境。';
      setErrorMessage(message);
      setPendingAssistantSessionId((current) => (current === currentSessionId ? null : current));
      setActiveRequestId(null);
      setAppState('idle');
      setProgressEvents((previous) =>
        upsertProgressEvent(previous, {
          requestId,
          stepId: 'error',
          label: '失败',
          status: 'error',
          detail: message,
          timestamp: new Date().toISOString(),
        }),
      );
    }
  };

  const handleOpenResult = (turn: AssistantSolveTurn) => {
    setActiveResult(turn.solveResponse);
    setAppState('result');
  };

  const handleBackToChat = () => {
    setAppState('idle');
  };

  const isBusy = activeRequestId !== null;

  return (
    <div className="flex h-screen overflow-hidden flex-col bg-[#f7f7f8] text-neutral-800 selection:bg-blue-500/30">
      <Header state={appState} onOpenSettings={() => setSettingsOpen(true)} />

      <main className="mx-auto grid min-h-0 w-full max-w-[1680px] flex-1 gap-6 overflow-hidden px-4 py-5 lg:grid-cols-[248px_minmax(0,1fr)] lg:px-6">
        <HistorySidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onCreateSession={handleCreateSession}
          onDeleteSession={handleDeleteSession}
        />

        <section className="flex min-h-0 flex-col overflow-hidden">
          {appState === 'solving' && (
            <div className="flex min-h-0 flex-1 overflow-y-auto rounded-[30px] border border-neutral-200 bg-white px-6 py-4 shadow-sm">
              <SolvingProcess events={progressEvents} />
            </div>
          )}

          {appState === 'result' && activeResult && (
            <ResultView solveResponse={activeResult} onBackToChat={handleBackToChat} onOpenNewChat={handleCreateSession} />
          )}

          {appState === 'idle' && (
            <div className="flex min-h-0 flex-1 flex-col">
              {currentSession.turns.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center overflow-y-auto">
                  <div className="w-full max-w-4xl">
                    <h1 className="text-center text-[2.2rem] font-medium tracking-tight text-neutral-900">你好，有什么我可以帮你的？</h1>

                    <div className="mt-8">
                      <InputArea
                        value={inputText}
                        mode={mode}
                        isBusy={isBusy}
                        isPickingFiles={isPickingFiles}
                        selectedFiles={attachmentViewModels}
                        onChange={setInputText}
                        onModeChange={setMode}
                        onPickFiles={handlePickFiles}
                        onRemoveSelectedFile={handleRemoveSelectedFile}
                        onSubmit={handleSend}
                      />
                    </div>

                    {errorMessage && (
                      <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{errorMessage}</div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex min-h-0 flex-1 flex-col rounded-[30px] border border-neutral-200 bg-white px-5 py-5 shadow-sm">
                  <div className="mb-4 border-b border-neutral-100 pb-4">
                    <div className="text-lg font-semibold text-neutral-900">{currentSession.title}</div>
                    <div className="mt-1 text-sm text-neutral-500">
                      {latestSessionResult
                        ? latestSessionResult.batchItems && latestSessionResult.batchItems.length > 1
                          ? `最近一次批量求解了 ${latestSessionResult.batchItems.length} 个实例。`
                          : `最近一次求解的最终距离为 ${latestSessionResult.result.final_solution.distance.toFixed(3).replace(/\.?0+$/, '')}。`
                        : `当前会话共有 ${currentSession.turns.length} 条消息。`}
                    </div>
                  </div>

                  <ChatThread
                    session={currentSession}
                    onOpenResult={handleOpenResult}
                    isAssistantLoading={pendingAssistantSessionId === currentSession.id}
                  />

                  <div className="mt-4 border-t border-neutral-100 pt-4">
                    <InputArea
                      value={inputText}
                      mode={mode}
                      isBusy={isBusy}
                      isPickingFiles={isPickingFiles}
                      selectedFiles={attachmentViewModels}
                      onChange={setInputText}
                      onModeChange={setMode}
                      onPickFiles={handlePickFiles}
                      onRemoveSelectedFile={handleRemoveSelectedFile}
                      onSubmit={handleSend}
                    />

                    {errorMessage && (
                      <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{errorMessage}</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <SettingsModal
        isOpen={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSaveSettings}
      />
    </div>
  );
}
