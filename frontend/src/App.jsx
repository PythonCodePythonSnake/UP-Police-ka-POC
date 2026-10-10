import React from 'react';
import { useComplaintSession } from './hooks/useComplaintSession';
import Header from './components/Header';
import Workspace from './components/Workspace';

export default function App() {
  const sessionData = useComplaintSession();

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col font-sans text-slate-900">
      <Header 
        onNewSession={sessionData.startNewSession} 
        hasSession={!!sessionData.sessionId} 
      />
      <Workspace sessionData={sessionData} />
    </div>
  );
}