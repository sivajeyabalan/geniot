import useAppStore from '../store'
import Panel from './ui/Panel'

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString()
}

function scoreClass(score) {
  if (score > 0.8) {
    return 'anomaly-badge high'
  }
  if (score > 0.5) {
    return 'anomaly-badge medium'
  }
  return 'anomaly-badge low'
}

export default function AnomalyPanel() {
  const anomalyEvents = useAppStore((state) => state.anomalyEvents)

  return (
    <Panel
      title="Recent Anomalies"
      subtitle="Detected events from VAE reconstruction scoring"
      className="anomaly-panel"
    >
      {anomalyEvents.length === 0 ? (
        <p className="anomaly-empty">No anomaly events detected yet.</p>
      ) : (
        <ul className="anomaly-list" aria-label="Anomaly events">
          {[...anomalyEvents].reverse().map((event, idx) => (
            <li key={`${event.timestamp}-${idx}`} className="anomaly-item">
              <time className="anomaly-time">{formatTime(event.timestamp)}</time>
              <span className={scoreClass(event.score)}>
                score {event.score.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}
