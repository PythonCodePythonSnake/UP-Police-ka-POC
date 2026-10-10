import React, { useState } from 'react';
import { ClipboardList, FileText } from 'lucide-react';
import ExtractedDetails from './ExtractedDetails';
import DraftReview from './DraftReview';

export default function ComplaintPanel({ sessionData }) {
  const [activeTab, setActiveTab] = useState('details');

  return (
    <div className="hidden lg:flex lg:col-span-5 xl:col-span-4 flex-col bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden h-[calc(100vh-7rem)]">
      
      {/* Tabs Header */}
      <div className="flex border-b border-slate-200 bg-slate-50 shrink-0">
        <button
          onClick={() => setActiveTab('details')}
          className={`flex-1 flex items-center justify-center gap-2 py-3.5 text-sm font-medium transition-colors ${
            activeTab === 'details' 
              ? 'bg-white text-blue-600 border-b-2 border-blue-600' 
              : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
          }`}
        >
          <ClipboardList className="w-4 h-4" />
          Extracted Details
        </button>
        <button
          onClick={() => setActiveTab('draft')}
          className={`flex-1 flex items-center justify-center gap-2 py-3.5 text-sm font-medium transition-colors ${
            activeTab === 'draft' 
              ? 'bg-white text-blue-600 border-b-2 border-blue-600' 
              : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
          }`}
        >
          <FileText className="w-4 h-4" />
          Complaint Draft
        </button>
      </div>

      {/* Tab Content Area */}
      <div className="flex-1 overflow-y-auto bg-slate-50/30">
        {activeTab === 'details' ? (
          <ExtractedDetails sessionData={sessionData} />
        ) : (
          <DraftReview sessionData={sessionData} />
        )}
      </div>
    </div>
  );
}