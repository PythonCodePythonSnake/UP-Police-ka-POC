import React, { useState, useEffect } from 'react';
import { FileText, Copy, RefreshCw, Check, AlertTriangle } from 'lucide-react';

export default function DraftReview({ sessionData }) {
  const { draft, status, error, generateDraft, complaint } = sessionData;
  const [localDraft, setLocalDraft] = useState('');
  const [copied, setCopied] = useState(false);

  // Sync the local editable textarea with the backend draft when it arrives
  useEffect(() => {
    if (draft) setLocalDraft(draft);
  }, [draft]);

  const handleCopy = () => {
    navigator.clipboard.writeText(localDraft);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Ensure they have provided at least one detail before allowing a draft
  const hasDetails = complaint && Object.keys(complaint.details || {}).length > 0;

  if (status === 'drafting') {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center text-slate-500">
        <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-4" />
        <p className="text-sm font-medium text-slate-700">Drafting your complaint...</p>
        <p className="text-xs mt-2 text-slate-400">The AI is structuring your incident details.</p>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-slate-50">
        <FileText className="w-12 h-12 text-slate-300 mb-4" />
        <h2 className="text-lg font-semibold text-slate-700 mb-2">Ready to Draft?</h2>
        <p className="text-sm text-slate-500 max-w-sm mb-6 leading-relaxed">
          Once you have provided enough details in the chat, generate a structured complaint draft to review and copy.
        </p>
        
        {error && status === 'error' && (
          <div className="mb-4 p-3 bg-red-50 text-red-700 text-xs rounded border border-red-200 max-w-sm text-left">
            <strong>Error:</strong> {error}
          </div>
        )}

        <button
          onClick={generateDraft}
          disabled={!hasDetails}
          className="bg-blue-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
        >
          Generate Draft
        </button>
        {!hasDetails && (
          <p className="text-xs text-slate-400 mt-3">Please provide some incident details in the chat first.</p>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Disclaimer Banner */}
      <div className="p-4 border-b border-slate-200 bg-amber-50 flex items-start gap-3 shrink-0">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-800 leading-relaxed">
          <strong>This is a preliminary draft for your review.</strong> It has <strong>not</strong> been officially filed or submitted to any police department. Please review and edit the text carefully before printing or using it.
        </div>
      </div>
      
      {/* Toolbar */}
      <div className="p-4 flex justify-between items-center bg-white border-b border-slate-100 shrink-0 shadow-sm z-10">
        <h3 className="text-sm font-bold text-slate-700">Review & Edit Draft</h3>
        <div className="flex gap-2">
          <button
            onClick={() => {
              if (window.confirm("Regenerate the draft? Your manual edits will be lost.")) {
                generateDraft();
              }
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-colors border border-slate-200"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Regenerate
          </button>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-800 hover:bg-slate-700 rounded-md transition-colors shadow-sm"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied!' : 'Copy Text'}
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 p-4 bg-slate-50/50 overflow-hidden flex flex-col">
        <textarea
          value={localDraft}
          onChange={(e) => setLocalDraft(e.target.value)}
          className="flex-1 w-full p-5 text-sm text-slate-800 bg-white border border-slate-200 rounded-lg shadow-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 resize-none font-serif leading-relaxed"
          spellCheck="false"
        />
      </div>
    </div>
  );
}