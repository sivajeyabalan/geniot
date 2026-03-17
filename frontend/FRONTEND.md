# Frontend — React Dashboard Instructions

> Copilot context file for `frontend/src/` directory.

---

## App overview

The React frontend is a **Digital Twin Dashboard** for the IoT network. It:
- Connects to the FastAPI backend via WebSocket for live data
- Displays a real-time network topology graph (React Flow)
- Shows live latency, throughput, and energy charts (Recharts)
- Highlights anomalies in a dedicated alert panel
- Allows triggering traffic generation and optimization via REST calls

---

## Pages

### Dashboard.jsx (main page)
- Top row: 4 metric cards (latency, throughput, energy, QoS)
- Middle left: `<NetworkTopology />` — React Flow network graph
- Middle right: `<TrafficChart />` — live line chart of metrics
- Bottom: `<AnomalyPanel />` — recent anomaly events list
- All fed by Zustand store updated by WebSocket

### Simulation.jsx
- Controls to run the GAN traffic generator
- Displays generated synthetic traffic sequences as a chart
- Compare real vs synthetic traffic distributions

### Optimization.jsx
- Input: current network state metrics (editable form)
- Button: "Run PPO Optimizer"
- Output: recommended configuration changes + expected improvement

---

## Component specs

### MetricsGrid.jsx
```jsx
// 4 metric cards in a CSS grid
// Props: { latency, throughput, energy, qos }
// Each card: label (top, muted), value (large, bold), delta badge (↑↓%)
// Colors: latency=blue, throughput=green, energy=amber, qos=purple
```

### TrafficChart.jsx
```jsx
// Recharts AreaChart with 60-point rolling window
// X axis: timestamp (formatted as HH:MM:SS)
// Y axes: latency (left, ms), throughput (right, Mbps)
// Two areas: latency (blue), throughput (green)
// Updates every 500ms from WebSocket data
// Props: { data: [{timestamp, latency, throughput, energy}] }
```

### NetworkTopology.jsx
```jsx
// React Flow graph of IoT nodes
// Node types: sensor (circle), gateway (square), server (diamond)
// Node colors: normal=gray, anomaly=red, optimized=green
// Edges: animated if data is flowing, dashed if link is degraded
// Props: { nodes: [...], edges: [...], anomalyNodes: [id,...] }
```

### AnomalyPanel.jsx
```jsx
// Scrollable list of recent anomaly events
// Each item: timestamp, anomaly score, affected node ID, type badge
// Badge types: "DDoS" (red), "Fault" (amber), "Congestion" (blue)
// Empty state: "No anomalies detected" with green checkmark
// Props: { events: [{id, timestamp, score, type, nodeId}] }
```

---

## Zustand store (store.js)

```javascript
import { create } from 'zustand'

const useStore = create((set, get) => ({
  // Live metrics (rolling 60-point window)
  metricsHistory: [],           // [{timestamp, latency, throughput, energy, qos}]
  addMetric: (metric) => set(state => ({
    metricsHistory: [...state.metricsHistory.slice(-59), metric]
  })),

  // Current snapshot
  currentMetrics: null,
  setCurrentMetrics: (m) => set({ currentMetrics: m }),

  // Network graph
  nodes: [],
  edges: [],
  setTopology: (nodes, edges) => set({ nodes, edges }),

  // Anomalies
  anomalyEvents: [],
  addAnomaly: (event) => set(state => ({
    anomalyEvents: [event, ...state.anomalyEvents].slice(0, 50)
  })),

  // Connection status
  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),

  // Generated traffic (Simulation page)
  syntheticTraffic: [],
  setSyntheticTraffic: (data) => set({ syntheticTraffic: data }),
}))

export default useStore
```

---

## WebSocket client (websocket.js)

```javascript
// Connects to ws://localhost:8000/ws/live-metrics
// Parses JSON: { timestamp, latency, throughput, energy, qos, anomaly, anomaly_score, nodes_active }
// On message:
//   1. store.addMetric(metric)
//   2. store.setCurrentMetrics(metric)
//   3. if metric.anomaly → store.addAnomaly({...})
// Reconnects automatically on disconnect (exponential backoff, max 30s)
```

---

## API calls (src/api.js)

```javascript
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

export const generateTraffic = (nSamples = 10) =>
  fetch(`${BASE_URL}/api/generate-traffic`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_samples: nSamples, seq_len: 50 })
  }).then(r => r.json())

export const detectAnomaly = (sequence) =>
  fetch(`${BASE_URL}/api/detect-anomaly`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ traffic_sequence: sequence })
  }).then(r => r.json())

export const optimize = (networkState) =>
  fetch(`${BASE_URL}/api/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ network_state: networkState })
  }).then(r => r.json())
```

---

## Environment variables (.env)

```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/live-metrics
VITE_REFRESH_MS=500
```

---

## Running frontend

```bash
cd frontend
npm install
npm run dev        # starts on http://localhost:5173
npm run build      # production build
```