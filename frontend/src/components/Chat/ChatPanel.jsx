import React, { useRef, useEffect } from 'react';
import { ShieldAlert, FileText, ArrowRight } from 'lucide-react';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';

export default function ChatPanel({ sessionData }) {
  const { sessionId, messages, startNewSession, sendMessage, status } = sessionData;
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new messages appear
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, status]);

  const isSending = status === 'sending';

  // Welcome Screen (No Active Session)
  if (!sessionId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-slate-50 overflow-y-auto">
        <ShieldAlert className="w-12 h-12 text-slate-400 mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-3">Start a New Complaint Draft</h2>
        <p className="text-sm text-slate-600 max-w-md mb-8 leading-relaxed">
          Describe the incident in your own words. The AI will guide you through collecting the necessary details to generate a structured, formal complaint draft.
        </p>
        <button
          onClick={startNewSession}
          disabled={status === 'loading'}
          className="flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-sm disabled:opacity-70"
        >
          {status === 'loading' ? 'Starting Session...' : 'Start Anonymous Session'}
          <ArrowRight className="w-4 h-4" />
        </button>
        
        <div className="mt-12 text-left max-w-md w-full">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 text-center">Example Prompts</p>
          <div className="space-y-3">
            <div className="text-sm bg-white border border-slate-200 p-4 rounded-lg text-slate-600 shadow-sm">
              "My phone was snatched at Kanpur Central yesterday evening by two men on a bike."
            </div>
            <div className="text-sm bg-white border border-slate-200 p-4 rounded-lg text-slate-600 shadow-sm">
              "I was defrauded online. Someone called claiming to be from my bank and took 20,000 rupees."
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Active Chat Screen
  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50">
      {/* Messages Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-400 space-y-3">
            <FileText className="w-8 h-8 opacity-30" />
            <p className="text-sm font-medium">Session started. Describe what happened.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
        
        {isSending && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border border-slate-200 px-4 py-4 rounded-2xl rounded-bl-none shadow-sm flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
              <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
              <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <MessageInput onSend={sendMessage} disabled={isSending || status === 'loading'} />
    </div>
  );
}