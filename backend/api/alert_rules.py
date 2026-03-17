"""Alert rules evaluation engine for live network metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


def _to_datetime(value: str) -> datetime:
    """Parse ISO timestamp into datetime."""
    return datetime.fromisoformat(value)


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}


@dataclass
class AlertRule:
    """Single threshold-based alert rule definition."""

    rule_id: str
    metric: str
    operator: str
    threshold: float
    duration_seconds: float
    severity: str
    enabled: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


class AlertRulesEngine:
    """Manage alert rules and evaluate them over streaming metrics."""

    def __init__(self) -> None:
        self.rules: dict[str, AlertRule] = {}
        self.alert_history: list[dict[str, Any]] = []
        self._condition_started_at: dict[str, str] = {}
        self._active_alerts: dict[str, bool] = {}

    def list_rules(self) -> list[dict[str, Any]]:
        """Return all rules sorted by creation time descending."""
        items = [self._rule_to_dict(rule) for rule in self.rules.values()]
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def create_rule(
        self,
        metric: str,
        operator: str,
        threshold: float,
        duration_seconds: float,
        severity: str,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create and store a new alert rule."""
        self._validate_rule_fields(metric, operator, duration_seconds, severity)

        rule_id = uuid4().hex
        rule = AlertRule(
            rule_id=rule_id,
            metric=metric,
            operator=operator,
            threshold=float(threshold),
            duration_seconds=float(duration_seconds),
            severity=severity,
            enabled=bool(enabled),
        )
        self.rules[rule_id] = rule
        self._active_alerts[rule_id] = False
        return self._rule_to_dict(rule)

    def update_rule(
        self,
        rule_id: str,
        metric: str,
        operator: str,
        threshold: float,
        duration_seconds: float,
        severity: str,
        enabled: bool,
    ) -> dict[str, Any]:
        """Update an existing rule by id."""
        if rule_id not in self.rules:
            raise KeyError(f"Rule not found: {rule_id}")

        self._validate_rule_fields(metric, operator, duration_seconds, severity)

        rule = self.rules[rule_id]
        rule.metric = metric
        rule.operator = operator
        rule.threshold = float(threshold)
        rule.duration_seconds = float(duration_seconds)
        rule.severity = severity
        rule.enabled = bool(enabled)
        rule.updated_at = utc_now_iso()

        self._condition_started_at.pop(rule_id, None)
        self._active_alerts[rule_id] = False
        return self._rule_to_dict(rule)

    def delete_rule(self, rule_id: str) -> None:
        """Delete rule and runtime state by id."""
        if rule_id not in self.rules:
            raise KeyError(f"Rule not found: {rule_id}")

        del self.rules[rule_id]
        self._condition_started_at.pop(rule_id, None)
        self._active_alerts.pop(rule_id, None)

    def evaluate(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate all enabled rules for one metric payload.

        Alerts are emitted only when the condition is sustained for the
        configured duration and only once per active violation window.
        """
        now_iso = payload.get("timestamp") or utc_now_iso()
        triggered: list[dict[str, Any]] = []

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                self._condition_started_at.pop(rule_id, None)
                self._active_alerts[rule_id] = False
                continue

            metric_value = payload.get(rule.metric)
            if metric_value is None:
                self._condition_started_at.pop(rule_id, None)
                self._active_alerts[rule_id] = False
                continue

            comparator = OPERATORS[rule.operator]
            condition_met = comparator(float(metric_value), float(rule.threshold))

            if not condition_met:
                self._condition_started_at.pop(rule_id, None)
                self._active_alerts[rule_id] = False
                continue

            start_iso = self._condition_started_at.get(rule_id)
            if start_iso is None:
                self._condition_started_at[rule_id] = now_iso
                start_iso = now_iso

            elapsed_seconds = (_to_datetime(now_iso) - _to_datetime(start_iso)).total_seconds()
            if elapsed_seconds < rule.duration_seconds:
                continue

            if self._active_alerts.get(rule_id, False):
                continue

            alert_event = {
                "id": uuid4().hex,
                "timestamp": now_iso,
                "rule_id": rule_id,
                "metric": rule.metric,
                "operator": rule.operator,
                "threshold": rule.threshold,
                "value": float(metric_value),
                "duration_seconds": rule.duration_seconds,
                "severity": rule.severity,
                "message": f"{rule.metric} {rule.operator} {rule.threshold} for {rule.duration_seconds:.1f}s",
            }
            self.alert_history.append(alert_event)
            self._active_alerts[rule_id] = True
            triggered.append(alert_event)

        return triggered

    def list_alert_history(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return recent alert history entries, newest first."""
        if limit <= 0:
            return []
        return list(reversed(self.alert_history[-limit:]))

    def _validate_rule_fields(
        self,
        metric: str,
        operator: str,
        duration_seconds: float,
        severity: str,
    ) -> None:
        """Validate supported fields for rules."""
        supported_metrics = {
            "latency_ms",
            "throughput_mbps",
            "energy_nj_per_bit",
            "qos_score",
            "anomaly_score",
            "nodes_active",
        }
        supported_severity = {"low", "medium", "high", "critical"}

        if metric not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        if operator not in OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")
        if float(duration_seconds) < 0:
            raise ValueError("duration_seconds must be >= 0")
        if severity.lower() not in supported_severity:
            raise ValueError(f"Unsupported severity: {severity}")

    @staticmethod
    def _rule_to_dict(rule: AlertRule) -> dict[str, Any]:
        """Serialize rule to JSON-compatible dict."""
        return {
            "id": rule.rule_id,
            "metric": rule.metric,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "duration_seconds": rule.duration_seconds,
            "severity": rule.severity,
            "enabled": rule.enabled,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at,
        }
