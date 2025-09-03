import axios from 'axios';

const API_BASE = 'https://max25-research_assistant_beckend.hf.space/api';

export const uploadFileToBackend = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE}/assistant/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data.summary;
};

export const generateChallengeQuestions = async (text: string) => {
  const formData = new FormData();
  formData.append('document_text', text);

  const response = await axios.post(`${API_BASE}/challenge/generate-questions`, formData);
  return response.data.questions;
};
