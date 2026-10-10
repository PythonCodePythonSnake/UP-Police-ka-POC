import axios from 'axios';

// Vite proxies '/api' to 'http://localhost:8000/api' as configured
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Session Management
  createSession: async () => {
    const response = await apiClient.post('/sessions');
    return response.data;
  },
  
  getSession: async (sessionId) => {
    const response = await apiClient.get(`/sessions/${sessionId}`);
    return response.data;
  },

  // Conversation
  sendMessage: async (sessionId, message) => {
    const response = await apiClient.post(`/chat/${sessionId}`, { message });
    return response.data;
  },

  // Complaint State
  getComplaint: async (sessionId) => {
    const response = await apiClient.get(`/complaints/${sessionId}`);
    return response.data;
  },

  updateComplaint: async (sessionId, details) => {
    const response = await apiClient.patch(`/complaints/${sessionId}`, { details });
    return response.data;
  },

  // Draft Generation
  generateDraft: async (sessionId) => {
    const response = await apiClient.post(`/complaints/${sessionId}/draft`);
    return response.data;
  }
};