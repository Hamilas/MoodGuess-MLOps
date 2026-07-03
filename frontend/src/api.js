const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5080'
const API_KEY = import.meta.env.VITE_API_KEY || 'MoodGuessDemo2024'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...options.headers,
    },
  })

  const body = await response.json().catch(() => null)

  if (!response.ok) {
    const message = body?.detail
      ? typeof body.detail === 'string'
        ? body.detail
        : JSON.stringify(body.detail)
      : `Request failed with status ${response.status}`
    throw new Error(message)
  }

  return body
}

export function getHealth() {
  return request('/health')
}

export function getReady() {
  return request('/ready')
}

export function predictSentiment(text) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function submitFeedback(predictionId, correctedLabel) {
  return request('/feedback', {
    method: 'POST',
    body: JSON.stringify({
      prediction_id: predictionId,
      corrected_label: correctedLabel,
    }),
  })
}

export function getModelInfo() {
  return request('/model-info')
}

export function getBusinessKpis() {
  return request('/monitoring/kpis/business')
}

export function getQualityKpis() {
  return request('/monitoring/kpis/quality')
}

export function getCostKpis() {
  return request('/monitoring/kpis/cost')
}

export function getPerformanceKpis() {
  return request('/monitoring/kpis/performance')
}

export function getSloStatus() {
  return request('/monitoring/kpis/slo')
}

export function getDriftSummary() {
  return request('/monitoring/drift')
}

export function checkDrift() {
  return request('/monitoring/drift/check')
}

export function getRegisteredModels() {
  return request('/monitoring/models')
}

export function submitBatchPrediction(texts, priority = 'medium') {
  return request('/batch/predict', {
    method: 'POST',
    body: JSON.stringify({ texts, priority }),
  })
}

export function getBatchStatus(jobId) {
  return request(`/batch/status/${jobId}`)
}

export function getBatchResults(jobId) {
  return request(`/batch/results/${jobId}`)
}

export function cancelBatchJob(jobId) {
  return request(`/batch/jobs/${jobId}`, { method: 'DELETE' })
}

export function getBatchQueueStatus() {
  return request('/batch/queue/status')
}
