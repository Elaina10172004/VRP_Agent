import React from 'react';
import { ArrowUp, Brain, CircleDashed, FileText, UploadCloud, X, Zap } from 'lucide-react';
import type { SolveMode } from '../types/solver';

type InputAreaProps = {
  value: string;
  mode: SolveMode;
  isBusy: boolean;
  isPickingFiles: boolean;
  selectedFileNames: string[];
  onChange: (value: string) => void;
  onModeChange: (mode: SolveMode) => void;
  onPickFiles: () => Promise<void> | void;
  onRemoveSelectedFile: (index: number) => void;
  onSubmit: () => void;
};

export function InputArea({
  value,
  mode,
  isBusy,
  isPickingFiles,
  selectedFileNames,
  onChange,
  onModeChange,
  onPickFiles,
  onRemoveSelectedFile,
  onSubmit,
}: InputAreaProps) {
  const canSubmit = Boolean(value.trim() || selectedFileNames.length > 0) && !isBusy;
  const isProcessingSelectedFiles = isBusy && selectedFileNames.length > 0;

  const handleSubmit = () => {
    if (canSubmit) {
      onSubmit();
    }
  };

  return (
    <div className="w-full">
      <div className="rounded-[28px] border border-neutral-200 bg-white shadow-sm transition-all duration-200 hover:border-neutral-300 focus-within:border-neutral-400 focus-within:shadow-md">
        {(selectedFileNames.length > 0 || isPickingFiles) && (
          <div className="px-4 pt-4">
            <div className="flex flex-wrap gap-2">
              {isPickingFiles && (
                <div className="inline-flex max-w-full items-center gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm text-neutral-700">
                  <CircleDashed size={16} className="shrink-0 animate-spin text-neutral-500" />
                  <span className="max-w-[220px] truncate">正在导入实例...</span>
                </div>
              )}
              {selectedFileNames.map((fileName, index) => (
                <div key={`${fileName}-${index}`} className="inline-flex max-w-full items-center gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm text-neutral-700">
                  {isProcessingSelectedFiles ? (
                    <CircleDashed size={16} className="shrink-0 animate-spin text-neutral-500" />
                  ) : (
                    <FileText size={16} className="shrink-0 text-neutral-500" />
                  )}
                  <span className="max-w-[220px] truncate">{fileName}</span>
                  <button
                    onClick={() => onRemoveSelectedFile(index)}
                    disabled={isBusy}
                    className="rounded-lg p-1 text-neutral-400 transition-colors hover:bg-white hover:text-neutral-900 disabled:cursor-not-allowed disabled:opacity-40"
                    title="移除附件"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder=""
          className="min-h-[96px] w-full resize-none bg-transparent px-5 py-4 text-base leading-relaxed text-neutral-900 outline-none placeholder:text-neutral-400"
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              handleSubmit();
            }
          }}
        />

        <div className="flex flex-col gap-3 border-t border-neutral-100 px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <button
              onClick={onPickFiles}
              disabled={isPickingFiles}
              className="inline-flex items-center gap-2 rounded-2xl px-3 py-2 text-sm text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-900 disabled:cursor-not-allowed disabled:opacity-50"
              title="上传实例文件"
            >
              {isPickingFiles ? <CircleDashed size={18} className="animate-spin" /> : <UploadCloud size={18} />}
              {isPickingFiles ? '导入中' : '上传文件'}
            </button>

            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="inline-flex items-center gap-2 rounded-2xl bg-black px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-35"
              title="发送"
            >
              发送
              <ArrowUp size={18} />
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => onModeChange('quick')}
              className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors ${
                mode === 'quick'
                  ? 'border-neutral-900 bg-neutral-900 text-white'
                  : 'border-neutral-200 bg-white text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <Zap size={16} className={mode === 'quick' ? 'text-amber-300' : ''} />
              快速
            </button>

            <button
              onClick={() => onModeChange('thinking')}
              className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors ${
                mode === 'thinking'
                  ? 'border-neutral-900 bg-neutral-900 text-white'
                  : 'border-neutral-200 bg-white text-neutral-500 hover:text-neutral-900'
              }`}
            >
              <Brain size={16} className={mode === 'thinking' ? 'text-sky-300' : ''} />
              思考
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
