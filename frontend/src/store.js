import { create } from 'zustand'

const MAX_METRICS_POINTS = 60
const MAX_ANOMALY_EVENTS = 50

const useAppStore = create((set) => ({
  metricsHistory: [],
  currentMetrics: null,
  anomalyEvents: [],
  wsConnected: false,

  addMetric: (metric) =>
    set((state) => ({
      metricsHistory: [...state.metricsHistory, metric].slice(-MAX_METRICS_POINTS),
    })),

  setCurrentMetrics: (metric) =>
    set(() => ({
      currentMetrics: metric,
    })),

  addAnomaly: (anomalyEvent) =>
    set((state) => ({
      anomalyEvents: [...state.anomalyEvents, anomalyEvent].slice(-MAX_ANOMALY_EVENTS),
    })),

  setWsConnected: (connected) =>
    set(() => ({
      wsConnected: connected,
    })),
}))

export default useAppStore
