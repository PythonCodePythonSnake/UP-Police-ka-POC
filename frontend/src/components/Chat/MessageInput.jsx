import React, { useState } from 'react';
import { Send } from 'lucide-react';

export default function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSend = () => {
    if (text.trim() && !disabled) {
      onSend(text.trim());
      setText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-4 bg-white border-t border-slate-200 shrink-0">
      <div className="flex items-end gap-3 bg-slate-50 border border-slate-300 rounded-xl p-2 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all">
        <textarea
          className="flex-1 max-h-32 bg-transparent resize-none outline-none text-sm p-2 text-slate-800 placeholder-slate-400"
          rows={2}
          placeholder="Describe what happened... (Press Enter to send, Shift+Enter for new line)"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="p-3 mb-0.5 rounded-lg bg-blue-600 text-white disabled:bg-slate-300 disabled:text-slate-500 hover:bg-blue-700 transition-colors shrink-0 shadow-sm"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
      <p className="text-[11px] text-slate-400 mt-3 text-center font-medium">
        This tool prepares a draft. It does not officially file a police complaint.
      </p>
    </div>
  );
}