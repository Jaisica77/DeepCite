// API service for backend integration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Document {
  doc_id: string;
  filename: string;
  file_type: string;
  text_length: number;
  word_count: number;
  status: string;
}

export interface ChunkMetadata {
  doc_id: string;
  chunk_id: number;
  chunk_size: number;
  text: string;
}

export interface RetrievedChunk {
  chunk: string;
  score: number;
  metadata: ChunkMetadata;
}

export interface RAGRequest {
  query: string;
  doc_ids: string[];
  chunk_size?: number;
  overlap_percent?: number;
  top_k?: number;
  model_name?: string;
  temperature?: number;
}

export interface RAGResponse {
  answer: string;
  query: string;
  retrieved_chunks: RetrievedChunk[];
  config: {
    chunk_size: number;
    overlap_percent: number;
    top_k: number;
    model: string;
    temperature: number;
  };
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  latency: number;
  total_chunks_indexed: number;
}

export interface ExperimentRequest {
  query: string;
  doc_ids: string[];
  chunk_sizes?: number[];
  overlap_percent?: number;
  top_k?: number;
  model_name?: string;
}

export interface ExperimentResponse {
  query: string;
  experiments: Array<{
    chunk_size: number;
    result?: RAGResponse;
    error?: string;
  }>;
  total_experiments: number;
}

export interface EvaluationRequest {
  query: string;
  generated_answer: string;
  expected_answer?: string;
  context_chunks?: string[];
  evaluator_model?: string;
}

export interface EvaluationResponse {
  scores: {
    relevance: number;
    accuracy: number;
    completeness: number;
    coherence: number;
    faithfulness: number;
    overall: number;
  };
  feedback: string;
  evaluator_model: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Citation {
  type: 'doc' | 'web';
  source: string;
  title?: string;
  snippet: string;
}

export interface AgentChatRequest {
  query: string;
  doc_ids?: string[];
  chat_history?: ChatMessage[];
  use_web_search?: boolean;
}

export interface AgentChatResponse {
  answer: string;
  citations: Citation[];
  tool_calls_made: number;
}

async function apiCall<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
    });
    if (!response.ok) {
      let errorDetail = response.statusText;
      try {
        const error = await response.json();
        errorDetail = error.detail || error.message || errorDetail;
      } catch {}
      throw new Error(errorDetail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        throw new Error('Network error: Failed to connect to server. Please check if the backend is running.');
      }
      throw error;
    }
    throw new Error('Unknown error occurred');
  }
}

export const uploadApi = {
  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/api/upload-docs`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `Upload failed: ${response.statusText}`);
    }
    return response.json();
  },
  listDocuments: () => apiCall<Document[]>('/api/docs'),
  getDocument: (doc_id: string) => apiCall<Document & { text_preview?: string }>(`/api/docs/${doc_id}`),
  deleteDocument: (doc_id: string) => apiCall<{ message: string; doc_id: string }>(`/api/docs/${doc_id}`, { method: 'DELETE' }),
};

export const ragApi = {
  runRAG: (request: RAGRequest) => apiCall<RAGResponse>('/api/run-rag', { method: 'POST', body: JSON.stringify(request) }),
  runExperiment: (request: ExperimentRequest) => apiCall<ExperimentResponse>('/api/run-experiment', { method: 'POST', body: JSON.stringify(request) }),
  getRetrieverStats: () => apiCall('/api/retriever-stats'),
  clearIndex: () => apiCall<{ message: string }>('/api/clear-index', { method: 'POST' }),
};

export const evaluationApi = {
  evaluate: (request: EvaluationRequest) => apiCall<EvaluationResponse>('/api/evaluate', { method: 'POST', body: JSON.stringify(request) }),
  comparePipelines: (query: string, results: any[]) => apiCall('/api/compare-pipelines', { method: 'POST', body: JSON.stringify({ query, results }) }),
  generateQuestions: (doc_id: string, num_questions = 5, model_name = 'gemini-1.5-flash') =>
    apiCall('/api/generate-questions', { method: 'POST', body: JSON.stringify({ doc_id, num_questions, model_name }) }),
  batchEvaluate: (queries: any[]) => apiCall('/api/batch-evaluate', { method: 'POST', body: JSON.stringify(queries) }),
};

export const agentApi = {
  chat: (request: AgentChatRequest) =>
    apiCall<AgentChatResponse>('/api/agent-chat', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
};

export const healthCheck = () => apiCall<{ status: string }>('/health');
