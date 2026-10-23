/**
 * INNOVEXA API Client Service
 * Communicates with FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return await res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE_URL}/models`);
  if (!res.ok) throw new Error(`Failed to load models: ${res.statusText}`);
  return await res.json();
}

export async function fetchDashboard() {
  const res = await fetch(`${API_BASE_URL}/dashboard`);
  if (!res.ok) throw new Error(`Failed to load dashboard: ${res.statusText}`);
  return await res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics`);
  if (!res.ok) throw new Error(`Failed to load metrics: ${res.statusText}`);
  return await res.json();
}

export async function fetchNetwork() {
  const res = await fetch(`${API_BASE_URL}/network`);
  if (!res.ok) throw new Error(`Failed to load network: ${res.statusText}`);
  return await res.json();
}

export async function fetchRobustness() {
  const res = await fetch(`${API_BASE_URL}/robustness`);
  if (!res.ok) throw new Error(`Failed to load robustness: ${res.statusText}`);
  return await res.json();
}

export async function fetchSamples() {
  const res = await fetch(`${API_BASE_URL}/samples`);
  if (!res.ok) throw new Error(`Failed to load samples: ${res.statusText}`);
  return await res.json();
}

export async function predictProfile(profileData) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(profileData)
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed (${res.status})`);
  }
  return await res.json();
}

export async function fetchHistory(limit = 50) {
  const res = await fetch(`${API_BASE_URL}/history?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to load history: ${res.statusText}`);
  return await res.json();
}

export async function clearHistoryApi() {
  const res = await fetch(`${API_BASE_URL}/history`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error(`Failed to clear history: ${res.statusText}`);
  return await res.json();
}
