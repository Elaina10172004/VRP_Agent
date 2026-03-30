import React, { useEffect, useRef, useState } from 'react';
import { MoreHorizontal, Plus, Trash2 } from 'lucide-react';
import type { ChatSession } from '../lib/history';

type HistorySidebarProps = {
  sessions: ChatSession[];
  currentSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (sessionId: string) => void;
};

export function HistorySidebar({ sessions, currentSessionId, onSelectSession, onCreateSession, onDeleteSession }: HistorySidebarProps) {
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);
  const menuRootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!menuRootRef.current?.contains(event.target as Node)) {
        setMenuSessionId(null);
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, []);

  return (
    <aside className="hidden min-h-0 bg-[#ececec] lg:flex lg:w-[252px] lg:flex-col">
      <div className="px-3 pb-3 pt-4">
        <button
          onClick={onCreateSession}
          className="flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-sm text-neutral-700 transition-colors hover:bg-white/70 hover:text-neutral-900"
        >
          <Plus size={16} />
          新对话
        </button>
      </div>

      <div ref={menuRootRef} className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <div className="space-y-1">
          {sessions.map((session) => {
            const isActive = session.id === currentSessionId;
            const isMenuOpen = menuSessionId === session.id;
            return (
              <div key={session.id} className="group relative">
                <button
                  onClick={() => {
                    setMenuSessionId(null);
                    onSelectSession(session.id);
                  }}
                  className={`w-full rounded-xl px-3 py-2.5 pr-10 text-left text-sm leading-6 transition-colors ${
                    isActive ? 'bg-white text-neutral-900' : 'text-neutral-700 hover:bg-white/70 hover:text-neutral-900'
                  }`}
                >
                  <span className="truncate block">{session.title || '新对话'}</span>
                </button>

                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    setMenuSessionId((current) => (current === session.id ? null : session.id));
                  }}
                  className={`absolute right-2 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-neutral-400 transition ${
                    isMenuOpen ? 'bg-white text-neutral-700' : 'opacity-0 group-hover:opacity-100 hover:bg-white hover:text-neutral-700'
                  }`}
                  title="更多"
                >
                  <MoreHorizontal size={16} />
                </button>

                {isMenuOpen && (
                  <div className="absolute right-2 top-[calc(100%-2px)] z-20 min-w-[120px] rounded-xl border border-neutral-200 bg-white p-1 shadow-lg">
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        setMenuSessionId(null);
                        onDeleteSession(session.id);
                      }}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-red-600 transition-colors hover:bg-red-50"
                    >
                      <Trash2 size={15} />
                      删除
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
