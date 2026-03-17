export default function StatusPill({ connected }) {
  return (
    <span
      className={`status-pill ${connected ? 'is-online' : 'is-offline'}`}
      role="status"
      aria-live="polite"
      aria-label={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
    >
      <span className="status-dot" aria-hidden="true" />
      {connected ? 'Connected' : 'Disconnected'}
    </span>
  )
}
