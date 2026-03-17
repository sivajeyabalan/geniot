import { useEffect } from 'react'
import useAppStore from './store'

const MAX_BACKOFF_MS = 30000

export function useWebSocket(url) {
  const addMetric = useAppStore((state) => state.addMetric)
  const setCurrentMetrics = useAppStore((state) => state.setCurrentMetrics)
  const addAnomaly = useAppStore((state) => state.addAnomaly)
  const setWsConnected = useAppStore((state) => state.setWsConnected)
  const isConnected = useAppStore((state) => state.wsConnected)

  useEffect(() => {
    let socket = null
    let reconnectTimeoutId = null
    let reconnectAttempts = 0
    let unmounted = false

    const connect = () => {
      if (unmounted) {
        return
      }

      socket = new WebSocket(url)

      socket.onopen = () => {
        reconnectAttempts = 0
        setWsConnected(true)
      }

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          const snapshot = {
            timestamp: message.timestamp,
            latency: Number(message.latency_ms ?? 0),
            throughput: Number(message.throughput_mbps ?? 0),
            energy: Number(message.energy_nj_per_bit ?? 0),
            qos: Number(message.qos_score ?? 0) * 100,
          }

          addMetric(snapshot)
          setCurrentMetrics(snapshot)

          if (message.anomaly_detected) {
            addAnomaly({
              timestamp: message.timestamp,
              score: Number(message.anomaly_score ?? 0),
            })
          }
        } catch (error) {
          console.error('WebSocket payload parse error', error)
        }
      }

      socket.onclose = () => {
        setWsConnected(false)
        if (unmounted) {
          return
        }

        reconnectAttempts += 1
        const delay = Math.min(1000 * 2 ** reconnectAttempts, MAX_BACKOFF_MS)
        reconnectTimeoutId = window.setTimeout(connect, delay)
      }

      socket.onerror = () => {
        if (socket) {
          socket.close()
        }
      }
    }

    connect()

    return () => {
      unmounted = true
      setWsConnected(false)
      if (reconnectTimeoutId) {
        window.clearTimeout(reconnectTimeoutId)
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close()
      }
    }
  }, [url, addMetric, setCurrentMetrics, addAnomaly, setWsConnected])

  return { isConnected }
}
