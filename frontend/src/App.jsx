import Dashboard from './pages/Dashboard'
import { useWebSocket } from './websocket'

const ROUTES = {
  '/': Dashboard,
  '/dashboard': Dashboard,
}

function App() {
  useWebSocket('ws://localhost:8000/ws/live-metrics')

  const pathname = window.location.pathname
  const ActivePage = ROUTES[pathname] || Dashboard

  return <ActivePage />
}

export default App
