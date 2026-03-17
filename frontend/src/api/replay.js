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
      // no-op
    }
    throw new Error(`${response.status}: ${detail}`)
  }

  return response.json()
}

export function listReplays() {
  return requestJson('/api/replay/list')
}

export function getReplay(id) {
  return requestJson(`/api/replay/${id}`)
}

export function startReplayRecording(payload = {}) {
  return requestJson('/api/replay/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function stopReplayRecording() {
  return requestJson('/api/replay/stop', {
    method: 'POST',
  })
}

export function getReplayStatus() {
  return requestJson('/api/replay/status/current')
}
