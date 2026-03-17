import { useEffect, useMemo, useRef, useState } from 'react'
import {
  getReplay,
  getReplayStatus,
  listReplays,
  startReplayRecording,
  stopReplayRecording,
} from '../api/replay'
import Panel from './ui/Panel'

const PLAYBACK_SPEEDS = [0.5, 1, 2, 4]

function toFrames(replay) {
  if (!replay?.events?.length) return []
  return replay.events
    .filter((event) => event.type === 'live-metric' && event.metrics)
    .map((event) => ({
      timestamp: event.timestamp,
      latency: Number(event.metrics.latency_ms ?? 0),
      throughput: Number(event.metrics.throughput_mbps ?? 0),
      energy: Number(event.metrics.energy_nj_per_bit ?? 0),
      anomaly: Boolean(event.metrics.anomaly_detected),
      score: Number(event.metrics.anomaly_score ?? 0),
    }))
}

function formatMetric(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) return `-- ${unit}`
  return `${value.toFixed(2)} ${unit}`
}

function ReplaySnapshot({ label, frame }) {
  return (
    <section className="replay-snapshot" aria-label={`${label} snapshot`}>
      <header className="replay-snapshot-header">
        <h3>{label}</h3>
        <time>{frame ? new Date(frame.timestamp).toLocaleTimeString() : '--:--:--'}</time>
      </header>
      <div className="replay-metric-grid">
        <p><span>Latency</span>{formatMetric(frame?.latency, 'ms')}</p>
        <p><span>Throughput</span>{formatMetric(frame?.throughput, 'Mbps')}</p>
        <p><span>Energy</span>{formatMetric(frame?.energy, 'nJ/bit')}</p>
        <p>
          <span>Anomaly</span>
          {frame ? (frame.anomaly ? `Yes (${frame.score.toFixed(2)})` : 'No') : '--'}
        </p>
      </div>
    </section>
  )
}

export default function ReplayStudio() {
  const [items, setItems] = useState([])
  const [status, setStatus] = useState({ recording: false })
  const [selectedA, setSelectedA] = useState('')
  const [selectedB, setSelectedB] = useState('')
  const [cache, setCache] = useState({})
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [position, setPosition] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const timerRef = useRef(null)

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const [listPayload, statusPayload] = await Promise.all([listReplays(), getReplayStatus()])
      const nextItems = listPayload.items || []
      setItems(nextItems)
      setStatus(statusPayload)

      if (!selectedA && nextItems[0]) {
        setSelectedA(nextItems[0].id)
      }
      if (!selectedB && nextItems[1]) {
        setSelectedB(nextItems[1].id)
      }
      if (!selectedB && !nextItems[1] && nextItems[0]) {
        setSelectedB(nextItems[0].id)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const ensureLoaded = async (id) => {
    if (!id || cache[id]) return cache[id]
    const replay = await getReplay(id)
    setCache((prev) => ({ ...prev, [id]: replay }))
    return replay
  }

  useEffect(() => {
    let mounted = true
    const loadSelected = async () => {
      try {
        if (selectedA) await ensureLoaded(selectedA)
        if (selectedB) await ensureLoaded(selectedB)
      } catch (err) {
        if (mounted) setError(err.message)
      }
    }
    loadSelected()
    return () => {
      mounted = false
    }
  }, [selectedA, selectedB])

  const framesA = useMemo(() => toFrames(cache[selectedA]), [cache, selectedA])
  const framesB = useMemo(() => toFrames(cache[selectedB]), [cache, selectedB])

  const maxFrames = Math.max(framesA.length, framesB.length, 1)
  const step = maxFrames > 1 ? 1 / (maxFrames - 1) : 1

  const frameA = useMemo(() => {
    if (!framesA.length) return null
    const idx = Math.min(framesA.length - 1, Math.floor(position * (framesA.length - 1)))
    return framesA[idx]
  }, [framesA, position])

  const frameB = useMemo(() => {
    if (!framesB.length) return null
    const idx = Math.min(framesB.length - 1, Math.floor(position * (framesB.length - 1)))
    return framesB[idx]
  }, [framesB, position])

  useEffect(() => {
    if (!playing) {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      return
    }

    timerRef.current = setInterval(() => {
      setPosition((prev) => {
        const next = prev + step
        if (next >= 1) {
          setPlaying(false)
          return 1
        }
        return next
      })
    }, Math.max(80, 500 / speed))

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [playing, speed, step])

  const onStartRecording = async () => {
    setError('')
    try {
      await startReplayRecording({ metadata: { source: 'dashboard-ui' } })
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const onStopRecording = async () => {
    setError('')
    try {
      await stopReplayRecording()
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const deltaLatency = frameA && frameB ? frameB.latency - frameA.latency : null
  const deltaThroughput = frameA && frameB ? frameB.throughput - frameA.throughput : null
  const deltaEnergy = frameA && frameB ? frameB.energy - frameA.energy : null

  return (
    <Panel
      title="Replay Studio"
      subtitle="Record, replay, and compare two sessions side-by-side"
      className="replay-panel"
    >
      <div className="replay-toolbar">
        <div className="replay-toolbar-group">
          <button type="button" className="btn" onClick={refresh} disabled={loading}>
            Refresh
          </button>
          <button type="button" className="btn" onClick={onStartRecording} disabled={status.recording}>
            Start Recording
          </button>
          <button type="button" className="btn" onClick={onStopRecording} disabled={!status.recording}>
            Stop Recording
          </button>
        </div>
        <p className={`replay-status ${status.recording ? 'recording' : ''}`}>
          {status.recording ? `Recording (${status.event_count || 0} events)` : 'Not recording'}
        </p>
      </div>

      <div className="replay-selectors">
        <label>
          Session A
          <select value={selectedA} onChange={(event) => setSelectedA(event.target.value)}>
            <option value="">Select replay</option>
            {items.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} ({item.event_count} events)
              </option>
            ))}
          </select>
        </label>

        <label>
          Session B
          <select value={selectedB} onChange={(event) => setSelectedB(event.target.value)}>
            <option value="">Select replay</option>
            {items.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} ({item.event_count} events)
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="replay-controls">
        <button type="button" className="btn" onClick={() => setPlaying((prev) => !prev)} disabled={maxFrames <= 1}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button type="button" className="btn" onClick={() => setPosition(0)}>
          Reset
        </button>
        <label className="replay-slider-label">
          Progress
          <input
            className="replay-slider"
            type="range"
            min={0}
            max={1000}
            value={Math.floor(position * 1000)}
            onChange={(event) => setPosition(Number(event.target.value) / 1000)}
          />
        </label>
        <label>
          Speed
          <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
            {PLAYBACK_SPEEDS.map((value) => (
              <option key={value} value={value}>
                {value}x
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="replay-compare-grid">
        <ReplaySnapshot label="Session A" frame={frameA} />
        <ReplaySnapshot label="Session B" frame={frameB} />
      </div>

      <div className="replay-delta-row" aria-live="polite">
        <p>Δ Latency: {deltaLatency === null ? '--' : `${deltaLatency.toFixed(2)} ms`}</p>
        <p>Δ Throughput: {deltaThroughput === null ? '--' : `${deltaThroughput.toFixed(2)} Mbps`}</p>
        <p>Δ Energy: {deltaEnergy === null ? '--' : `${deltaEnergy.toFixed(2)} nJ/bit`}</p>
      </div>

      {error ? <p className="replay-error">{error}</p> : null}
    </Panel>
  )
}
