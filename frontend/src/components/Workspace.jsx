import React from 'react';
import ChatPanel from './Chat/ChatPanel';
import ComplaintPanel from './Summary/ComplaintPanel';

export default function Workspace({ sessionData }) {
  return (
    <main className="flex-1 overflow-hidden p-4 md:p-6 max-w-[1600px] mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      {/* Chat Column */}
      <div className="lg:col-span-7 xl:col-span-8 flex flex-col bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden h-[calc(100vh-7rem)]">
        <ChatPanel sessionData={sessionData} />
      </div>

      {/* Complaint Summary Column */}
      <ComplaintPanel sessionData={sessionData} />
      
    </main>
  );
}