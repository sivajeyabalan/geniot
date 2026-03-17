import AnomalyPanel from '../components/AnomalyPanel'
import MetricsGrid from '../components/MetricsGrid'
import ReplayStudio from '../components/ReplayStudio'
import TrafficChart from '../components/TrafficChart'
import StatusPill from '../components/ui/StatusPill'
import useAppStore from '../store'

export default function Dashboard() {
  const wsConnected = useAppStore((state) => state.wsConnected)

  return (
    <main className="dashboard-page" aria-labelledby="dashboard-title">
      <header className="dashboard-header">
        <div>
          <h1 id="dashboard-title" className="dashboard-title">
            GenIoT-Optimizer
          </h1>
          <p className="dashboard-subtitle">Real-time IoT network monitoring and optimization</p>
        </div>
        <StatusPill connected={wsConnected} />
      </header>

      <MetricsGrid />
      <TrafficChart />
      <ReplayStudio />
      <AnomalyPanel />
    </main>
  )
}
