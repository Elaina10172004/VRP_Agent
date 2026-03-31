import type { ConversationTurn, DesktopSolveResponse, DesktopUploadedFileRef, SolveMode } from '../types/solver';

export type UserTurn = {
  id: string;
  role: 'user';
  createdAt: string;
  text: string;
  uploadedFileNames: string[];
  mode: SolveMode;
};

export type AssistantReplyTurn = {
  id: string;
  role: 'assistant';
  kind: 'reply';
  createdAt: string;
  text: string;
};

export type AssistantSolveTurn = {
  id: string;
  role: 'assistant';
  kind: 'solve';
  createdAt: string;
  solveResponse: DesktopSolveResponse;
};

export type AssistantTurn = AssistantReplyTurn | AssistantSolveTurn;
export type ChatTurn = UserTurn | AssistantTurn;

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  agentPreviousResponseId: string | null;
  turns: ChatTurn[];
};

const HISTORY_STORAGE_KEY = 'optichat:chat-history';
const ACTIVE_SESSION_STORAGE_KEY = 'optichat:active-session';

function hasWindow(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

export function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createEmptySession(title = '新对话'): ChatSession {
  const now = new Date().toISOString();
  return {
    id: createId('session'),
    title,
    createdAt: now,
    updatedAt: now,
    agentPreviousResponseId: null,
    turns: [],
  };
}

export function ensureSessions(sessions: ChatSession[]): ChatSession[] {
  return sessions.length > 0 ? sessions : [createEmptySession()];
}

function normalizeSession(input: unknown): ChatSession | null {
  if (!input || typeof input !== 'object') {
    return null;
  }

  const session = input as Record<string, unknown>;
  const turns = Array.isArray(session.turns) ? session.turns : [];
  return {
    id: typeof session.id === 'string' ? session.id : createId('session'),
    title: typeof session.title === 'string' && session.title.trim() ? session.title : '新对话',
    createdAt: typeof session.createdAt === 'string' ? session.createdAt : new Date().toISOString(),
    updatedAt: typeof session.updatedAt === 'string' ? session.updatedAt : new Date().toISOString(),
    agentPreviousResponseId:
      typeof session.agentPreviousResponseId === 'string' && session.agentPreviousResponseId.trim()
        ? session.agentPreviousResponseId
        : null,
    turns: turns
      .map((turn) => {
        if (!turn || typeof turn !== 'object') {
          return null;
        }

        const record = turn as Record<string, unknown>;
        if (record.role === 'user') {
          const legacyName =
            typeof record.uploadedFileName === 'string' && record.uploadedFileName.trim() ? [record.uploadedFileName.trim()] : [];
          return {
            id: typeof record.id === 'string' ? record.id : createId('user'),
            role: 'user' as const,
            createdAt: typeof record.createdAt === 'string' ? record.createdAt : new Date().toISOString(),
            text: typeof record.text === 'string' ? record.text : '',
            uploadedFileNames: Array.isArray(record.uploadedFileNames)
              ? record.uploadedFileNames.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
              : legacyName,
            mode: record.mode === 'thinking' ? 'thinking' : 'quick',
          };
        }

        if (record.role === 'assistant' && record.kind === 'solve' && record.solveResponse) {
          return {
            id: typeof record.id === 'string' ? record.id : createId('assistant-solve'),
            role: 'assistant' as const,
            kind: 'solve' as const,
            createdAt: typeof record.createdAt === 'string' ? record.createdAt : new Date().toISOString(),
            solveResponse: record.solveResponse as DesktopSolveResponse,
          };
        }

        return {
          id: typeof record.id === 'string' ? record.id : createId('assistant-reply'),
          role: 'assistant' as const,
          kind: 'reply' as const,
          createdAt: typeof record.createdAt === 'string' ? record.createdAt : new Date().toISOString(),
          text: typeof record.text === 'string' ? record.text : '',
        };
      })
      .filter((turn): turn is ChatTurn => turn !== null),
  };
}

export function loadChatSessions(): ChatSession[] {
  if (!hasWindow()) {
    return [createEmptySession()];
  }

  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) {
      return [createEmptySession()];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [createEmptySession()];
    }
    return ensureSessions(parsed.map((session) => normalizeSession(session)).filter((session): session is ChatSession => session !== null));
  } catch {
    return [createEmptySession()];
  }
}

export function saveChatSessions(sessions: ChatSession[]): void {
  if (!hasWindow()) {
    return;
  }
  window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(sessions));
}

export function loadActiveSessionId(): string | null {
  if (!hasWindow()) {
    return null;
  }
  return window.localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
}

export function saveActiveSessionId(sessionId: string): void {
  if (!hasWindow()) {
    return;
  }
  window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
}

function resolveAgentPreviousResponseId(nextValue: string | null | undefined, currentValue: string | null): string | null {
  if (nextValue === undefined) {
    return currentValue ?? null;
  }
  return nextValue;
}

function createUserTurn(input: { text: string; uploadedFiles?: DesktopUploadedFileRef[] | null; mode: SolveMode }, createdAt: string): UserTurn {
  return {
    id: createId('user'),
    role: 'user',
    createdAt,
    text: input.text,
    uploadedFileNames: input.uploadedFiles?.map((file) => file.name).filter(Boolean) ?? [],
    mode: input.mode,
  };
}

export function appendUserTurn(
  session: ChatSession,
  input: { text: string; uploadedFiles?: DesktopUploadedFileRef[] | null; mode: SolveMode },
): ChatSession {
  const now = new Date().toISOString();
  const userTurn = createUserTurn(input, now);

  return {
    ...session,
    updatedAt: now,
    turns: [...session.turns, userTurn],
  };
}

export function appendAssistantReplyMessage(
  session: ChatSession,
  message: string,
  agentPreviousResponseId?: string | null,
): ChatSession {
  const now = new Date().toISOString();
  const assistantTurn: AssistantReplyTurn = {
    id: createId('assistant-reply'),
    role: 'assistant',
    kind: 'reply',
    createdAt: now,
    text: message,
  };

  return {
    ...session,
    updatedAt: now,
    agentPreviousResponseId: resolveAgentPreviousResponseId(agentPreviousResponseId, session.agentPreviousResponseId),
    turns: [...session.turns, assistantTurn],
  };
}

export function appendAssistantSolveMessage(
  session: ChatSession,
  solveResponse: DesktopSolveResponse,
  agentPreviousResponseId?: string | null,
): ChatSession {
  const now = new Date().toISOString();
  const assistantTurn: AssistantSolveTurn = {
    id: createId('assistant-solve'),
    role: 'assistant',
    kind: 'solve',
    createdAt: now,
    solveResponse,
  };

  return {
    ...session,
    updatedAt: now,
    agentPreviousResponseId: resolveAgentPreviousResponseId(agentPreviousResponseId, session.agentPreviousResponseId),
    turns: [...session.turns, assistantTurn],
  };
}

export function appendSolveTurn(
  session: ChatSession,
  input: { text: string; uploadedFiles?: DesktopUploadedFileRef[] | null; mode: SolveMode },
  solveResponse: DesktopSolveResponse,
  agentPreviousResponseId?: string | null,
): ChatSession {
  return appendAssistantSolveMessage(appendUserTurn(session, input), solveResponse, agentPreviousResponseId);
}

export function appendReplyTurn(
  session: ChatSession,
  input: { text: string; uploadedFiles?: DesktopUploadedFileRef[] | null; mode: SolveMode },
  message: string,
  agentPreviousResponseId?: string | null,
): ChatSession {
  return appendAssistantReplyMessage(appendUserTurn(session, input), message, agentPreviousResponseId);
}

export function getLatestSolveResponse(session: ChatSession | null | undefined): DesktopSolveResponse | null {
  if (!session) {
    return null;
  }

  for (let index = session.turns.length - 1; index >= 0; index -= 1) {
    const turn = session.turns[index];
    if (turn.role === 'assistant' && turn.kind === 'solve') {
      return turn.solveResponse;
    }
  }
  return null;
}

export function buildConversationContext(session: ChatSession, limit = 8): ConversationTurn[] {
  return session.turns.slice(-limit).map((turn) => {
    if (turn.role === 'user') {
      const parts: string[] = [];
      if (turn.uploadedFileNames.length > 0) {
        parts.push(`附件：${turn.uploadedFileNames.join('，')}`);
      }
      if (turn.text.trim()) {
        parts.push(turn.text.trim());
      }
      return {
        role: 'user',
        content: parts.join('\n') || '用户上传了附件，但没有输入额外文字。',
      };
    }

    if (turn.kind === 'reply') {
      return {
        role: 'assistant',
        content: turn.text,
      };
    }

    if (turn.solveResponse.batchItems && turn.solveResponse.batchItems.length > 1) {
      return {
        role: 'assistant',
        content: `我已经完成一次批量求解，共处理 ${turn.solveResponse.batchItems.length} 个实例。`,
      };
    }

    const problemType = turn.solveResponse.result.problem_type.toUpperCase();
    const distance = turn.solveResponse.result.final_solution.distance.toFixed(3).replace(/\.?0+$/, '');
    return {
      role: 'assistant',
      content: `我已经完成一次 ${problemType} 求解，当前最终距离为 ${distance}。`,
    };
  });
}
