# Home Routing Audit & Hardening (Package R-01)

This audit documents the end-to-end trace, route precedence rules, edge cases, verified defects, fixes, and regression guarantees for the six-choice Home routing system in DCLab.

---

## 1. End-to-End Traced Flow

1. **User Action (Frontend):**
   - In [home.html](file:///home/antigravity/Shahriyar-Moradi/Research_Development_DCLab/dclab_rnd/agentic/web/src/views/home.html), the user clicks one of six `.help-kind` buttons (`Model` / `prediction`, `Plan` / `business`, `Report` / `reporting`, `Analyze` / `analysis`, `Prepare` / `preparation`, `Learn` / `advice`).
   - If a button is clicked while already active, it is toggled off (`on = false`).
   - The client invokes `DC.client.chooseIntent(draft_id, { body: { track: on ? hk.dataset.help : null } })`.

2. **API Endpoint:**
   - In [dclab_rnd/draft/api.py](file:///home/antigravity/Shahriyar-Moradi/Research_Development_DCLab/dclab_rnd/draft/api.py) at `POST /api/drafts/{draft_id}/intent`:
     - If `track in (None, "")`, calls `agent(draft_id).clear_intent(draft_id)`.
     - Otherwise validates `track in intents.TRACKS` and calls `agent(draft_id).choose_intent(draft_id, track)`.

3. **Storage & Agent Update:**
   - For `CaseHome` ([case_home.py](file:///home/antigravity/Shahriyar-Moradi/Research_Development_DCLab/dclab_rnd/draft/case_home.py)):
     - Updates `draft["intent"] = intents.chosen(track, before)` with `source="user"`.
     - Sets `case_state["chip_intent"] = None` so the press is always picked up.
     - On clearing (`clear_intent`), sets `draft["intent"] = None`, `case_state["chip"] = None`, and `case_state["chip_intent"] = None`.
   - For `HomeAgent` ([chat.py](file:///home/antigravity/Shahriyar-Moradi/Research_Development_DCLab/dclab_rnd/draft/chat.py)):
     - Updates `draft["intent"]` and moves the draft to that track without executing unprompted work.

4. **Lead Agent Execution ([case_agent.py](file:///home/antigravity/Shahriyar-Moradi/Research_Development_DCLab/dclab_rnd/agents/case_agent.py)):**
   - `CaseState` receives `chip`, `chip_intent`, and `chip_checked`.
   - In `turn()`, `chip_note(state)` injects a directive into the context.
   - If user words conflict on Turn 1, the agent is directed to propose both once (`chip_checked = True`).
   - If user words explicitly change mind on Turn 2+ (`changed_mind(state)`), the latest user words govern.

5. **Client Refresh & UI Synchronization:**
   - In `drawIntent(intent)`, each `.help-kind` button's `aria-pressed` attribute is synchronized with `Boolean(intent && intent.source === 'user' && b.dataset.help === intent.track)`.
   - On page refresh, active user chips persist accurately in the DOM.

---

## 2. Route Precedence Table

| Precedence | Scenario | Active Route / Action | Explanation |
|---|---|---|---|
| **1 (Highest)** | Explicit Change of Mind (Turn 2+) | Latest user choice in words | Words matching `CHANGE_WORDS` (e.g. "actually", "instead", "rather") update `state.chip` to the newly requested track. |
| **2** | Explicit Chip Selection (No Conflict) | Selected chip track | Selected chip outranks open inference. Agent starts or proposes inside the chosen track. |
| **3** | Chip vs. Turn 1 Text Contradiction | Surfaced once as 2 options | When user clicks e.g. `Learn` (advice) but types "predict churn on telco.csv", agent proposes exactly two options (one `advice`, one `prediction`) once; user selection resolves it. |
| **4** | Deselected Chip (`track: null`) | Inferred from words/data | Clicking an active chip clears user intent; system falls back to normal inference without erasing chat history. |
| **5 (Lowest)** | Absent Selection (Default) | Inference (`route_check` + rules) | Home uses `route_reading` / `route_check` and `C.named_tracks` to determine best track. |

---

## 3. Six Choices & Invariant Validation

1. **Model (`prediction`):**
   - Shows ML Solution workflow, outcome, and moment panels.
   - Demands table upload if `state.table is None` rather than falling back to advice.
   - Does NOT auto-trigger model training or build; build requires explicit user quote/confirmation.

2. **Plan (`business`):**
   - Strictly business planning/decisions. Does NOT conflict with or execute hierarchical task planning (HTP).

3. **Report (`reporting`):**
   - Dashboard and summary reports. Demands table if shared data is missing.

4. **Analyze (`analysis`):**
   - Queries, segment breakdowns, and distributions. Demands table if shared data is missing.

5. **Prepare (`preparation`):**
   - Data cleaning, dedup, transformations. Demands table if shared data is missing.

6. **Learn (`advice`):**
   - Strictly for conceptual questions, explanations, and methodologies.
   - Blocked from starting when `data_work(state)` is detected.

---

## 4. Confirmed Defects & Resolutions

1. **Defect 1: Chip Deselection Failed to Clear on Server**
   - *Problem:* Clicking an active chip toggled `aria-pressed` locally, but did not notify the server, leaving `draft["intent"]` and `state.chip` active.
   - *Fix:* In `home.html`, the click handler now sends `track: on ? hk.dataset.help : null`. In `draft/api.py`, `track in (None, "")` calls `agent(draft_id).clear_intent(draft_id)`, which resets intent in both `CaseHome` and `HomeAgent`.

2. **Defect 2: Active Chip Lost on Browser Refresh**
   - *Problem:* `drawIntent(intent)` updated intent text labels but did not update `.help-kind` `aria-pressed` attributes. On page refresh, all chip buttons reverted to `aria-pressed="false"`.
   - *Fix:* In `drawIntent(intent)`, `el.querySelectorAll('.help-kind').forEach(b => b.setAttribute('aria-pressed', String(Boolean(intent && intent.source === 'user' && b.dataset.help === intent.track))))` was added.

3. **Defect 3: Missing Table Caused Downgrade to Advice**
   - *Problem:* When a user requested prediction, analysis, reporting, or preparation without uploading a CSV first, the agent frequently downgraded to `advice`.
   - *Fix:* In `case_agent.py`:
     - `data_work(state)` recognizes data work from words or chips even when `state.table is None`.
     - `_start(state, track, tool)` refuses data work without a table with `"they want {track}: ask them to share or upload the table it runs on, then start it"`.
     - `_start(state, "advice", ...)` refuses to start advice when `data_work(state)` is True.

4. **Defect 4: Clear Request Trap on Turn 1**
   - *Problem:* When Turn 1 provided complete goal, unit/target, and data, Lead Agent asked clarifying questions instead of acting.
   - *Fix:* Pure helper `is_clear_on_arrival(case, table)` in `dclab_core/case.py`, runtime `clear_note` directive in `case_agent.py::turn()`, and deterministic refusal in `_ask` on clear Turn 1 arrival.

---

## 5. Security & Isolation Guarantees

- **No Execution During Routing:** A selected chip only sets intent metadata; it never initiates background training, project builds, or tool executions.
- **Holdout & Data Integrity:** Routing does not read raw table values or leak sensitive columns.
- **Offline / No-Model Compatibility:** All precedence logic is deterministic and fully verifiable without live LLM calls.
