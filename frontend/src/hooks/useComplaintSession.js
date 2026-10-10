import { useState, useCallback } from 'react';
import { api } from '../api';

export function useComplaintSession() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [complaint, setComplaint] = useState(null);
  const [draft, setDraft] = useState(null);
  const [status, setStatus] = useState('idle'); // 'idle' | 'loading' | 'sending' | 'drafting' | 'error'
  const [error, setError] = useState(null);

  // Helper to safely execute API calls and update state
  const execute = async (operation, loadingState = 'loading') => {
    setStatus(loadingState);
    setError(null);
    try {
      const result = await operation();
      setStatus('idle');
      return result;
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An unexpected error occurred.');
      setStatus('error');
      throw err;
    }
  };

  const startNewSession = useCallback(async () => {
    const data = await execute(api.createSession);
    setSessionId(data.session_id);
    setMessages([]);
    setComplaint(null);
    setDraft(null);
    return data.session_id;
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!sessionId) return;
    
    // Optimistically add user message to UI
    const tempUserMessage = { role: 'user', content: text, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, tempUserMessage]);

    await execute(async () => {
      const result = await api.sendMessage(sessionId, text);
      // Backend returns the exact assistant reply and the updated complaint state
      setMessages((prev) => [...prev, result.reply]);
      setComplaint({
        details: result.complaint.details,
        allegations: result.complaint.allegations,
        missingFields: result.missing_required_fields
      });
    }, 'sending');
  }, [sessionId]);

  const updateDetail = useCallback(async (field, value) => {
    if (!sessionId) return;
    await execute(async () => {
      const result = await api.updateComplaint(sessionId, {
        [field]: { value, uncertain: false }
      });
      setComplaint({
        details: result.complaint.details,
        allegations: result.complaint.allegations,
        missingFields: result.missing_required_fields
      });
    });
  }, [sessionId]);

  const generateDraft = useCallback(async () => {
    if (!sessionId) return;
    await execute(async () => {
      const result = await api.generateDraft(sessionId);
      setDraft(result.draft + '\n\n' + result.disclaimer);
    }, 'drafting');
  }, [sessionId]);

  return {
    sessionId,
    messages,
    complaint,
    draft,
    status,
    error,
    startNewSession,
    sendMessage,
    updateDetail,
    generateDraft
  };
}