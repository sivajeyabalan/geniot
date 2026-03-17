# Sprint Tickets — GenIoT-Optimizer

This file converts the roadmap into build-ready tickets with effort and clear acceptance checks.

## Effort Scale
- **S**: 0.5–1 day
- **M**: 1–3 days
- **L**: 3–5 days
- **XL**: 5+ days

---

## Sprint 1 (High Impact Demo Value)

### TKT-001 — Scenario Replay Backend
- **Priority:** P0
- **Effort:** M
- **Owner:** Backend
- **Description:** Add replay recording and retrieval endpoints for live metrics/actions/anomalies.
- **Tasks:**
  - [ ] Create replay session model (`id`, `start`, `end`, `metadata`)
  - [ ] Persist streamed snapshots to `results/replays/<id>.json`
  - [ ] Implement endpoints:
    - [ ] `POST /api/replay/start`
    - [ ] `POST /api/replay/stop`
    - [ ] `GET /api/replay/list`
    - [ ] `GET /api/replay/{id}`
- **Acceptance Criteria:**
  - [ ] Can record at least 5 minutes continuously
  - [ ] Replay file is deterministic and loads correctly

### TKT-002 — Scenario Replay Frontend
- **Priority:** P0
- **Effort:** M
- **Owner:** Frontend
- **Description:** Add replay player controls and baseline-vs-GenIoT comparison mode.
- **Tasks:**
  - [ ] Add replay selector UI
  - [ ] Add controls: play/pause/seek/speed
  - [ ] Add side-by-side metric cards and chart comparison
  - [ ] Add delta badges for latency/throughput/energy/F1
- **Acceptance Criteria:**
  - [ ] Replay playback is smooth with no UI freeze
  - [ ] Comparison view updates in sync across both panes

### TKT-003 — Alert Rules Engine (Backend)
- **Priority:** P0
- **Effort:** M
- **Owner:** Backend
- **Description:** Build threshold rule evaluator and alert generation.
- **Tasks:**
  - [ ] Define rule schema: metric/operator/threshold/duration/severity
  - [ ] Evaluate rules against live stream
  - [ ] Persist alert history
  - [ ] Implement endpoints:
    - [ ] `GET /api/rules`
    - [ ] `POST /api/rules`
    - [ ] `PUT /api/rules/{id}`
    - [ ] `DELETE /api/rules/{id}`
- **Acceptance Criteria:**
  - [ ] Alerts trigger only when rule duration condition is met
  - [ ] Rule CRUD works without server restart

### TKT-004 — Alert Rules UI
- **Priority:** P0
- **Effort:** M
- **Owner:** Frontend
- **Description:** Add rule management and active alerts panel.
- **Tasks:**
  - [ ] Rules create/edit/delete form
  - [ ] Active alerts list with severity filter
  - [ ] Alert acknowledgement action
- **Acceptance Criteria:**
  - [ ] New rule appears and triggers in live stream
  - [ ] Acknowledged alert state persists

### TKT-005 — Topology View with React Flow
- **Priority:** P1
- **Effort:** M
- **Owner:** Frontend + Backend
- **Description:** Visualize live nodes/links with health state and anomaly highlights.
- **Tasks:**
  - [ ] Backend payload for nodes/edges/health
  - [ ] React Flow rendering component
  - [ ] Dynamic styles by node/link status
  - [ ] Hover details tooltip
- **Acceptance Criteria:**
  - [ ] Topology updates at stream cadence
  - [ ] Anomalous nodes are visually distinct

### TKT-006 — One-Click Demo Mode
- **Priority:** P1
- **Effort:** S
- **Owner:** DevOps/App
- **Description:** Add scripts to start backend+frontend+optional scenario in one command.
- **Tasks:**
  - [ ] Add `run_all.bat`
  - [ ] Add optional scenario seed argument
  - [ ] Add stop script (`stop_all.bat`)
- **Acceptance Criteria:**
  - [ ] Demo stack boots with one command
  - [ ] Demo run is reproducible

---

## Sprint 2 (Production Readiness)

### TKT-007 — Manual Override + Guardrails
- **Priority:** P1
- **Effort:** M
- **Owner:** Full Stack
- **Description:** Allow operator control override with safety limits and rollback.
- **Tasks:**
  - [ ] Add backend validation for action bounds
  - [ ] Add override endpoint and rollback endpoint
  - [ ] Add frontend override panel and rollback button
- **Acceptance Criteria:**
  - [ ] Out-of-range actions rejected with clear error
  - [ ] Rollback restores prior config immediately

### TKT-008 — Persistent Metrics Storage
- **Priority:** P1
- **Effort:** L
- **Owner:** Backend
- **Description:** Persist metrics and expose historical query APIs.
- **Tasks:**
  - [ ] DB schema for metrics/events
  - [ ] Stream writer service
  - [ ] Add history endpoints with range/interval params
- **Acceptance Criteria:**
  - [ ] Query for last 24h < 2s response
  - [ ] Data survives service restart

### TKT-009 — Historical Dashboard UI
- **Priority:** P1
- **Effort:** M
- **Owner:** Frontend
- **Description:** Add historical charts and CSV export.
- **Tasks:**
  - [ ] Date range picker
  - [ ] Historical chart mode
  - [ ] CSV export button
- **Acceptance Criteria:**
  - [ ] Historical data loads accurately for selected range
  - [ ] Export file matches chart data

### TKT-010 — Policy A/B Test Harness
- **Priority:** P1
- **Effort:** M
- **Owner:** ML/Backend
- **Description:** Compare policies under seeded scenarios and auto-generate report.
- **Tasks:**
  - [ ] Seeded runner
  - [ ] Baseline/GenIoT comparison computation
  - [ ] JSON/CSV report output
- **Acceptance Criteria:**
  - [ ] Same seed => reproducible outputs
  - [ ] Report includes latency/throughput/energy/F1 deltas

### TKT-011 — Explainability Panel
- **Priority:** P2
- **Effort:** M
- **Owner:** Full Stack
- **Description:** Show confidence and top contributing signals for anomaly/optimizer decisions.
- **Tasks:**
  - [ ] Add backend explanation payload
  - [ ] Add frontend explainability cards
- **Acceptance Criteria:**
  - [ ] Each decision includes confidence score and top contributors

### TKT-012 — Nightly Auto-Tuning Jobs
- **Priority:** P2
- **Effort:** L
- **Owner:** ML/Backend
- **Description:** Scheduled retraining/eval and threshold recalibration.
- **Tasks:**
  - [ ] Add scheduled job runner
  - [ ] Version model artifacts
  - [ ] Regression alerting when metrics drop
- **Acceptance Criteria:**
  - [ ] Nightly run produces artifacts + summary report
  - [ ] Regression threshold breach triggers alert

### TKT-013 — Auth + RBAC
- **Priority:** P2
- **Effort:** L
- **Owner:** Full Stack
- **Description:** Add JWT auth and role-based restrictions.
- **Tasks:**
  - [ ] Login endpoint + token issue
  - [ ] Viewer/Admin role checks
  - [ ] Frontend route guards
- **Acceptance Criteria:**
  - [ ] Viewer cannot modify rules/configs
  - [ ] Admin can access all controls

---

## Suggested Assignment
- **Frontend-focused:** TKT-002, TKT-004, TKT-005, TKT-009, TKT-011
- **Backend-focused:** TKT-001, TKT-003, TKT-006, TKT-008, TKT-010, TKT-012, TKT-013
- **ML-focused:** TKT-010, TKT-011, TKT-012

---

## Sprint Exit Criteria
- [ ] No critical errors during 5-minute continuous run
- [ ] All P0 tickets completed and demoable
- [ ] Docs updated (`projectdescription.md`, `howtorun.md`)
- [ ] API endpoints documented and testable
