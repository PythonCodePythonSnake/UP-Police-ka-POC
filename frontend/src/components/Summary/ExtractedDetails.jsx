import React, { useState } from 'react';
import { CheckCircle2, AlertCircle, Edit2, X, Check, FileQuestion, ClipboardList } from 'lucide-react';
const FIELD_LABELS = {
  complainant_name: "Complainant Name",
  complainant_contact: "Contact Details",
  incident_description: "Incident Description",
  incident_datetime: "Date & Time",
  incident_location: "Location",
  complaint_type: "Complaint Type",
  accused_details: "Accused Details",
  loss_or_harm: "Loss or Harm",
  evidence_witnesses: "Evidence & Witnesses"
};

// Inline Edit Component for a single field
function EditableField({ fieldKey, label, detail, isMissing, onSave }) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempValue, setTempValue] = useState(detail?.value || '');

  const handleSave = () => {
    if (tempValue.trim() !== (detail?.value || '')) {
      onSave(fieldKey, tempValue.trim());
    }
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <div className="bg-white p-3 rounded-lg border border-blue-300 shadow-sm mb-3">
        <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1.5 block">
          {label}
        </label>
        <textarea
          autoFocus
          className="w-full text-sm p-2 border border-slate-200 rounded bg-slate-50 focus:outline-none focus:border-blue-500 resize-none"
          rows={2}
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
        />
        <div className="flex justify-end gap-2 mt-2">
          <button onClick={() => setIsEditing(false)} className="p-1.5 text-slate-400 hover:text-slate-600 bg-slate-100 rounded">
            <X className="w-4 h-4" />
          </button>
          <button onClick={handleSave} className="p-1.5 text-white bg-blue-600 hover:bg-blue-700 rounded shadow-sm">
            <Check className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="group bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm mb-3 hover:border-blue-200 transition-colors relative">
      <div className="flex justify-between items-start mb-1">
        <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
          {label}
        </label>
        {!isMissing && (
          <button 
            onClick={() => {
              setTempValue(detail?.value || '');
              setIsEditing(true);
            }}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-blue-600 p-1"
          >
            <Edit2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {isMissing ? (
        <div className="flex items-center gap-2 text-slate-400 text-sm mt-1 py-1">
          <FileQuestion className="w-4 h-4" />
          <span className="italic">Not yet provided</span>
        </div>
      ) : (
        <div>
          <p className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">{detail.value}</p>
          <div className="flex items-center gap-1.5 mt-2.5 pt-2.5 border-t border-slate-50">
            {detail.uncertain ? (
              <span className="flex items-center gap-1 text-[11px] font-medium text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200">
                <AlertCircle className="w-3 h-3" />
                Needs Confirmation
              </span>
            ) : (
              <span className="flex items-center gap-1 text-[11px] font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                <CheckCircle2 className="w-3 h-3" />
                Recorded
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ExtractedDetails({ sessionData }) {
  const { complaint, updateDetail } = sessionData;

  if (!complaint) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8 text-slate-500">
        <ClipboardList className="w-10 h-10 text-slate-300 mb-4" />
        <p className="text-sm max-w-[200px]">As you chat, the AI will extract the facts of your incident and display them here.</p>
      </div>
    );
  }

  const { details = {}, allegations = [], missingFields = [] } = complaint;

  return (
    <div className="p-4 md:p-6 pb-20">
      
      {/* Known Details */}
      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 ml-1">Recorded Facts</h3>
      {Object.entries(details).map(([key, detail]) => (
        <EditableField 
          key={key} 
          fieldKey={key} 
          label={FIELD_LABELS[key] || key} 
          detail={detail} 
          onSave={updateDetail}
        />
      ))}

      {/* Missing Required Fields */}
      {missingFields.length > 0 && (
        <>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mt-8 mb-4 ml-1">Required Information Missing</h3>
          {missingFields.map((key) => (
            <EditableField 
              key={key} 
              fieldKey={key} 
              label={FIELD_LABELS[key] || key} 
              isMissing={true}
              onSave={updateDetail}
            />
          ))}
        </>
      )}

      {/* Allegations */}
      {allegations.length > 0 && (
        <div className="mt-8">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 ml-1">Allegations</h3>
          <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
            <ul className="divide-y divide-slate-100">
              {allegations.map((allegation, idx) => (
                <li key={idx} className="p-3.5 text-sm text-slate-800 flex items-start gap-3 bg-slate-50/50">
                  <span className="mt-0.5 text-slate-400 font-medium text-xs">{idx + 1}.</span>
                  <span className="leading-relaxed">{allegation}</span>
                </li>
              ))}
            </ul>
          </div>
          <p className="text-[11px] text-slate-400 mt-2 ml-1">Allegations are kept exactly as reported.</p>
        </div>
      )}
    </div>
  );
}