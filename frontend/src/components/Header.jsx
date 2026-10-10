import React from 'react';
import { Shield, PlusCircle } from 'lucide-react';

export default function Header({ onNewSession, hasSession }) {
  return (
    <header className="bg-white border-b border-slate-200 h-16 flex items-center justify-between px-6 shrink-0 shadow-sm z-10">
      <div className="flex items-center gap-3">
        <Shield className="w-6 h-6 text-slate-700" />
        <div>
          <h1 className="font-semibold text-slate-900 leading-tight">AI Complaint Assistant</h1>
          <p className="text-[11px] text-slate-500 font-medium uppercase tracking-wide">Proof of Concept</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        {hasSession && (
          <span className="text-xs font-medium bg-slate-100 text-slate-600 px-3 py-1 rounded-full border border-slate-200">
            Anonymous Session
          </span>
        )}
        <button
          onClick={() => {
            if (!hasSession || window.confirm("Start a new complaint? Unsaved chat will be lost.")) {
              onNewSession();
            }
          }}
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors"
        >
          <PlusCircle className="w-4 h-4" />
          New Complaint
        </button>
      </div>
    </header>
  );
}