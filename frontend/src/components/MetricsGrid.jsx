import { memo, useMemo } from 'react'
import useAppStore from '../store'
import Panel from './ui/Panel'

function formatMetric(value, suffix, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return `-- ${suffix}`
  }
  return `${Number(value).toFixed(digits)} ${suffix}`
}

const MetricCard = memo(function MetricCard({ label, current, previous, suffix, invertTrend = false }) {
  const improving = previous === null || previous === undefined
    ? null
    : invertTrend
      ? current >= previous
      : current <= previous

  const badgeClass = improving === null ? 'metric-badge neutral' : improving ? 'metric-badge good' : 'metric-badge bad'
  const badgeText = improving === null ? 'stable' : improving ? 'improving' : 'worsening'

  return (
    <article className="metric-card" aria-label={`${label} metric`}>
      <p className="metric-label">{label}</p>
      <p className="metric-value">{formatMetric(current, suffix)}</p>
      <span className={badgeClass}>{badgeText}</span>
    </article>
  )
})

export default function MetricsGrid() {
  const currentMetrics = useAppStore((state) => state.currentMetrics)
  const metricsHistory = useAppStore((state) => state.metricsHistory)

  const previous = useMemo(
    () => (metricsHistory.length > 1 ? metricsHistory[metricsHistory.length - 2] : null),
    [metricsHistory],
  )

  return (
    <Panel
      title="Network Metrics"
      subtitle="Live snapshot updates from the simulation stream"
      className="metrics-panel"
    >
      <section className="metrics-grid" aria-label="Live network metrics">
        <MetricCard
          label="Latency"
          current={currentMetrics?.latency}
          previous={previous?.latency}
          suffix="ms"
          invertTrend={false}
        />
        <MetricCard
          label="Throughput"
          current={currentMetrics?.throughput}
          previous={previous?.throughput}
          suffix="Mbps"
          invertTrend={true}
        />
        <MetricCard
          label="Energy"
          current={currentMetrics?.energy}
          previous={previous?.energy}
          suffix="nJ/bit"
          invertTrend={false}
        />
        <MetricCard
          label="QoS"
          current={currentMetrics?.qos}
          previous={previous?.qos}
          suffix="%"
          invertTrend={true}
        />
      </section>
    </Panel>
  )
}
