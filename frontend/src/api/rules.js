const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch {
      detail = response.statusText
    }
    throw new Error(`${response.status}: ${detail}`)
  }

  return response.json()
}

export function listRules() {
  return requestJson('/api/rules')
}

export function createRule(payload) {
  return requestJson('/api/rules', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function updateRule(ruleId, payload) {
  return requestJson(`/api/rules/${ruleId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  })
}

export function deleteRule(ruleId) {
  return requestJson(`/api/rules/${ruleId}`, {
    method: 'DELETE',
  })
}
