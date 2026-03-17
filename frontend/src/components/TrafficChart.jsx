import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useMemo } from 'react'
import useAppStore from '../store'
import Panel from './ui/Panel'

export default function TrafficChart() {
  const metricsHistory = useAppStore((state) => state.metricsHistory)

  const chartData = useMemo(
    () =>
      metricsHistory.map((item) => ({
        ...item,
        time: new Date(item.timestamp).toLocaleTimeString(),
      })),
    [metricsHistory],
  )

  return (
    <Panel title="Traffic Trend" subtitle="Rolling 60-point window" className="chart-panel">
      <div className="chart-container" role="img" aria-label="Latency and throughput area chart over time">
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
            <XAxis dataKey="time" tick={{ fontSize: 12 }} minTickGap={24} axisLine={false} tickLine={false} />
            <YAxis yAxisId="latency" orientation="left" axisLine={false} tickLine={false} width={52} />
            <YAxis yAxisId="throughput" orientation="right" axisLine={false} tickLine={false} width={52} />
            <Tooltip />
            <Area
              yAxisId="latency"
              type="monotone"
              dataKey="latency"
              stroke="#ef4444"
              fill="#fecaca"
              fillOpacity={0.45}
              name="Latency (ms)"
              isAnimationActive
            />
            <Area
              yAxisId="throughput"
              type="monotone"
              dataKey="throughput"
              stroke="#2563eb"
              fill="#bfdbfe"
              fillOpacity={0.45}
              name="Throughput (Mbps)"
              isAnimationActive
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}
