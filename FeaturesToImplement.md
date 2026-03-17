# Features To Implement

This file lists the next practical features to build for GenIoT-Optimizer, with direct value and minimal fluff.

## Priority Order (Recommended)
1. Scenario Replay + Compare
2. Alert Rules Engine
3. Topology View (React Flow)
4. One-Click Demo Mode
5. Manual Override + Safety Guardrails
6. Persistent Metrics Store + Query API
7. Policy A/B Testing Harness
8. Model Confidence + Explainability Panel
9. Auto-Tuning Jobs (Nightly)
10. Auth + Role-Based Access

---

## 1) Scenario Replay + Compare
**Goal:** Record a full run and replay it; compare Baseline vs GenIoT side-by-side.

**Why it matters:** Best feature for demos, debugging, and proving model benefit.

**Backend tasks:**
- Add recorder for streamed metrics/actions/anomalies.
- Save replay files (JSON/CSV) with timestamps and metadata.
- Add endpoints:
  - `POST /api/replay/start`
  - `POST /api/replay/stop`
  - `GET /api/replay/list`
  - `GET /api/replay/{id}`

**Frontend tasks:**
- Replay controls (play/pause/seek/speed).
- Split view cards/charts: Baseline vs GenIoT.
- “Diff” badges for latency/throughput/energy/F1.

**Acceptance criteria:**
- Can record >= 5 minutes and replay deterministically.
- Side-by-side comparison view works without page refresh.

---

## 2) Alert Rules Engine
**Goal:** User-defined threshold rules trigger alerts and optional webhooks.

**Why it matters:** Operational value beyond visualization.

**Backend tasks:**
- Rule schema (metric, operator, threshold, duration, severity).
- Rule evaluator over live stream.
- Optional webhook emitter.
- Endpoints:
  - `GET /api/rules`
  - `POST /api/rules`
  - `PUT /api/rules/{id}`
  - `DELETE /api/rules/{id}`

**Frontend tasks:**
- Rules CRUD UI.
- Active alerts panel with severity and timestamps.
- Alert acknowledgment and filter by severity.

**Acceptance criteria:**
- Rules trigger reliably for sustained conditions.
- Alert history visible and exportable.

---

## 3) Topology View (React Flow)
**Goal:** Visualize node/link health and anomaly hotspots live.

**Why it matters:** Strong visual impact and intuitive network understanding.

**Backend tasks:**
- Stream topology payload (nodes, edges, node health, link quality).
- Include anomaly flags per node.

**Frontend tasks:**
- Add `NetworkTopology` component using React Flow.
- Node styles by status: normal/warning/anomaly/optimized.
- Edge styles by link quality and traffic direction.
- Hover tooltips with node metrics.

**Acceptance criteria:**
- Graph updates in near real-time.
- Anomaly nodes are instantly visible.

---

## 4) One-Click Demo Mode
**Goal:** Single command/script to run backend+frontend and launch a scripted scenario.

**Why it matters:** Removes setup friction; reproducible demos.

**Implementation tasks:**
- Create startup script (`run_all.bat` and optional shell equivalent).
- Start backend, frontend, and scenario driver.
- Optional “Demo timeline” that injects predictable anomalies/events.

**Acceptance criteria:**
- One command starts complete demo stack.
- Same storyline reproducible in every presentation.

---

## 5) Manual Override + Safety Guardrails
**Goal:** Human can override optimizer actions with safe bounds and rollback.

**Why it matters:** Production trust and safety.

**Backend tasks:**
- Validate control bounds server-side.
- Add rollback endpoint to restore previous config.

**Frontend tasks:**
- Override controls for routing/sleep/power/buffer.
- Guardrail indicators + validation hints.
- “Rollback to last stable config” action.

**Acceptance criteria:**
- Unsafe values are rejected.
- Rollback works immediately and is logged.

---

## 6) Persistent Metrics Store + Query API
**Goal:** Save live stream data and expose history endpoints.

**Why it matters:** Enables trend analytics and reporting.

**Backend tasks:**
- Persist metrics to DB (PostgreSQL preferred).
- Add query endpoints:
  - `GET /api/history?from=&to=&interval=`
  - `GET /api/events?from=&to=`
- Add retention policy and indexing.

**Frontend tasks:**
- Date-range picker and historical charts.
- Export to CSV.

**Acceptance criteria:**
- Historical queries return correct aggregated data quickly.

---

## 7) Policy A/B Testing Harness
**Goal:** Compare two policies under identical seeded scenarios.

**Why it matters:** Objective performance proof.

**Backend tasks:**
- Runner for seeded episodes.
- Compare metrics and confidence intervals.
- Save experiment summary.

**Frontend tasks:**
- A/B config panel.
- Result table + chart.

**Acceptance criteria:**
- Same seed yields reproducible comparison output.
- Report generated automatically.

---

## 8) Model Confidence + Explainability Panel
**Goal:** Show confidence and rationale for anomaly flags and optimizer actions.

**Why it matters:** Reduces “black-box” concerns.

**Backend tasks:**
- Return confidence scores and top contributing features.
- Add lightweight explanation payload in optimize/anomaly endpoints.

**Frontend tasks:**
- Explainability panel with ranked feature contributions.
- Confidence badges and uncertainty hints.

**Acceptance criteria:**
- Each decision includes confidence and explanation metadata.

---

## 9) Auto-Tuning Jobs (Nightly)
**Goal:** Scheduled retraining/evaluation to avoid model drift.

**Why it matters:** Maintains performance over time.

**Backend tasks:**
- Nightly job runner for retrain/eval.
- Automatic VAE threshold recalibration.
- Save artifacts + performance report.

**Acceptance criteria:**
- Job runs unattended and stores versioned artifacts.
- Alert if metrics regress beyond threshold.

---

## 10) Auth + Role-Based Access
**Goal:** Basic auth with roles: admin vs viewer.

**Why it matters:** Required for production-like deployment.

**Backend tasks:**
- JWT auth and user roles.
- Protect write/critical endpoints.

**Frontend tasks:**
- Login page and route protection.
- Role-based UI controls.

**Acceptance criteria:**
- Viewer cannot change configs/rules.
- Admin can manage all controls.

---

## Suggested 2-Week Delivery Plan
**Week 1:**
- Scenario Replay + Compare
- Alert Rules Engine
- Topology View

**Week 2:**
- One-Click Demo Mode
- Manual Override + Guardrails
- Persistent Metrics Store (phase 1)

---

## Definition of Done (Global)
- Feature has backend endpoint(s), frontend UI, and testable acceptance criteria.
- No console/runtime errors during 5-minute continuous run.
- Included in docs and demo flow.
