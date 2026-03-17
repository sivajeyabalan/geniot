import { useEffect, useMemo, useState } from 'react'
import { createRule, deleteRule, listRules, updateRule } from '../api/rules'
import useAppStore from '../store'
import Panel from './ui/Panel'

const METRIC_OPTIONS = [
  'latency_ms',
  'throughput_mbps',
  'energy_nj_per_bit',
  'qos_score',
  'anomaly_score',
  'nodes_active',
]

const OPERATOR_OPTIONS = ['>', '>=', '<', '<=', '==', '!=']
const SEVERITY_OPTIONS = ['low', 'medium', 'high', 'critical']

function buildFormFromRule(rule) {
  return {
    metric: rule.metric,
    operator: rule.operator,
    threshold: String(rule.threshold),
    durationSeconds: String(rule.duration_seconds),
    severity: rule.severity,
    enabled: Boolean(rule.enabled),
  }
}

function defaultForm() {
  return {
    metric: 'latency_ms',
    operator: '>',
    threshold: '80',
    durationSeconds: '3',
    severity: 'medium',
    enabled: true,
  }
}

export default function AlertRulesPanel() {
  const [rules, setRules] = useState([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [editingId, setEditingId] = useState('')
  const [severityFilter, setSeverityFilter] = useState('all')
  const [form, setForm] = useState(defaultForm)

  const alertEvents = useAppStore((state) => state.alertEvents)
  const setAlertEvents = useAppStore((state) => state.setAlertEvents)
  const acknowledgedAlertIds = useAppStore((state) => state.acknowledgedAlertIds)
  const acknowledgeAlert = useAppStore((state) => state.acknowledgeAlert)

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const payload = await listRules()
      setRules(payload.items || [])
      if (Array.isArray(payload.alert_history)) {
        setAlertEvents(payload.alert_history)
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

  const visibleAlerts = useMemo(() => {
    const ordered = [...alertEvents].sort((left, right) => {
      return new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime()
    })
    if (severityFilter === 'all') {
      return ordered
    }
    return ordered.filter((item) => item.severity === severityFilter)
  }, [alertEvents, severityFilter])

  const onChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  const onSubmit = async (event) => {
    event.preventDefault()
    setSaving(true)
    setError('')

    const payload = {
      metric: form.metric,
      operator: form.operator,
      threshold: Number(form.threshold),
      duration_seconds: Number(form.durationSeconds),
      severity: form.severity,
      enabled: Boolean(form.enabled),
    }

    try {
      if (editingId) {
        await updateRule(editingId, payload)
      } else {
        await createRule(payload)
      }
      setForm(defaultForm())
      setEditingId('')
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  const onEdit = (rule) => {
    setEditingId(rule.id)
    setForm(buildFormFromRule(rule))
  }

  const onCancelEdit = () => {
    setEditingId('')
    setForm(defaultForm())
  }

  const onDelete = async (ruleId) => {
    setError('')
    try {
      await deleteRule(ruleId)
      if (editingId === ruleId) {
        onCancelEdit()
      }
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const onToggleEnabled = async (rule) => {
    setError('')
    try {
      await updateRule(rule.id, {
        metric: rule.metric,
        operator: rule.operator,
        threshold: Number(rule.threshold),
        duration_seconds: Number(rule.duration_seconds),
        severity: rule.severity,
        enabled: !rule.enabled,
      })
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <Panel
      title="Alert Rules"
      subtitle="Create threshold rules and monitor active alerts"
      className="rules-panel"
    >
      <div className="rules-toolbar">
        <button type="button" className="btn" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <form className="rules-form" onSubmit={onSubmit}>
        <label>
          Metric
          <select value={form.metric} onChange={(event) => onChange('metric', event.target.value)}>
            {METRIC_OPTIONS.map((metric) => (
              <option key={metric} value={metric}>{metric}</option>
            ))}
          </select>
        </label>

        <label>
          Operator
          <select value={form.operator} onChange={(event) => onChange('operator', event.target.value)}>
            {OPERATOR_OPTIONS.map((operator) => (
              <option key={operator} value={operator}>{operator}</option>
            ))}
          </select>
        </label>

        <label>
          Threshold
          <input
            type="number"
            step="0.01"
            value={form.threshold}
            onChange={(event) => onChange('threshold', event.target.value)}
            required
          />
        </label>

        <label>
          Duration (s)
          <input
            type="number"
            step="0.1"
            min="0"
            value={form.durationSeconds}
            onChange={(event) => onChange('durationSeconds', event.target.value)}
            required
          />
        </label>

        <label>
          Severity
          <select value={form.severity} onChange={(event) => onChange('severity', event.target.value)}>
            {SEVERITY_OPTIONS.map((severity) => (
              <option key={severity} value={severity}>{severity}</option>
            ))}
          </select>
        </label>

        <label className="rules-checkbox-label">
          <input
            type="checkbox"
            checked={form.enabled}
            onChange={(event) => onChange('enabled', event.target.checked)}
          />
          Enabled
        </label>

        <div className="rules-actions">
          <button type="submit" className="btn" disabled={saving}>
            {saving ? 'Saving...' : editingId ? 'Update Rule' : 'Create Rule'}
          </button>
          {editingId ? (
            <button type="button" className="btn" onClick={onCancelEdit}>
              Cancel
            </button>
          ) : null}
        </div>
      </form>

      <div className="rules-list">
        <h3>Configured Rules</h3>
        {rules.length === 0 ? (
          <p className="rules-empty">No rules configured yet.</p>
        ) : (
          <ul>
            {rules.map((rule) => (
              <li key={rule.id} className="rule-item">
                <div>
                  <strong>{rule.metric}</strong> {rule.operator} {rule.threshold} for {rule.duration_seconds}s
                  <div className="rule-meta">severity: {rule.severity} · {rule.enabled ? 'enabled' : 'disabled'}</div>
                </div>
                <div className="rule-item-actions">
                  <button type="button" className="btn" onClick={() => onEdit(rule)}>Edit</button>
                  <button type="button" className="btn" onClick={() => onToggleEnabled(rule)}>
                    {rule.enabled ? 'Disable' : 'Enable'}
                  </button>
                  <button type="button" className="btn" onClick={() => onDelete(rule.id)}>Delete</button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="alerts-section">
        <div className="alerts-header-row">
          <h3>Active Alerts</h3>
          <label>
            Severity Filter
            <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
              <option value="all">all</option>
              {SEVERITY_OPTIONS.map((severity) => (
                <option key={severity} value={severity}>{severity}</option>
              ))}
            </select>
          </label>
        </div>

        {visibleAlerts.length === 0 ? (
          <p className="rules-empty">No alerts for selected severity.</p>
        ) : (
          <ul className="alerts-list">
            {visibleAlerts.map((alert) => {
              const acknowledged = Boolean(acknowledgedAlertIds[alert.id])
              return (
                <li key={alert.id} className="alert-item">
                  <div>
                    <time>{new Date(alert.timestamp).toLocaleTimeString()}</time>
                    <div className="rule-meta">{alert.message}</div>
                  </div>
                  <div className="rule-item-actions">
                    <span className={`alert-severity ${alert.severity}`}>{alert.severity}</span>
                    <button
                      type="button"
                      className="btn"
                      disabled={acknowledged}
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      {acknowledged ? 'Acknowledged' : 'Acknowledge'}
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {error ? <p className="replay-error">{error}</p> : null}
    </Panel>
  )
}
