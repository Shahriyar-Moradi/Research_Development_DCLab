# Plan: one memory for DCLab's agents (Phase 11, packages 35.x)

Written 2026-10-27. Plan only: no code is built from it until the owner says so.

## Why

DCLab's agents remember some things today, each in its own place, and forget the rest:

- **Project notes** (`dclab_rnd/studio/memory.py`): decisions and corrections on one project, the latest 20 read by the
  intern and the orchestrator (`memory.summary`, injected in `intern/loop.py`, `intern/tools.py`,
  `draft/orchestrator.py`).
- **Lessons** (`dclab_rnd/lessons.py`): proposed after a project, reviewed by a person, then searched as evidence
  (`lessons.merged`). They never reach a prompt directly.
- **Episodes**: the experience log (`experience/`) and the decision ledger (`ledger/`): everything that happened, kept
  whole, searched by the Admin pages, never read by an agent.
- **Shared knowledge**: the evidence index (`evidence/knowledge/`, generated), searched by `retrieval/`.
- **DCLab Code** (the separate app) reads `CLAUDE.md` files through the Agent SDK and R&D's rules live, and keeps no
  memory of its own.

What is missing: facts about a **person** or a **workspace** (the owner wants simple answers, Persian text in an HTML
page), one shape for every kind of memory, a way for an agent to **ask** its memory, rules for **forgetting**, and a
place to **see and correct** what is remembered.

Two facts checked on 2026-10-27 shape the order:

1. **There is no vector database yet.** `retrieval/` keeps keyword search (BM25) and word vectors computed in memory
   (LSA, not a neural embedding). Its latest measurement (RET-011, 60 questions) found **0 of 10 paraphrased
   questions** with every method; overall recall at 5 is 0.71 (keywords) and 0.72 (hybrid). Neural embeddings were
   measured only with OpenAI's model, now blocked (`DCLAB_BLOCK_PROVIDERS=openai`); Anthropic offers no embeddings.
2. **Nothing is installed for it locally.** Ollama has no models; the local PostgreSQL 16 has no `vector` extension.

So vectors are measured first (35.0), and the structure (35.1–35.4) does not depend on them: a vector is an optional
index on a memory item, never the memory itself.

## The structure

### Five kinds

| Kind | Holds | DCLab example | Today |
|---|---|---|---|
| `working` | the conversation under way | the Telco chat right now | the agent's own messages; never stored as memory |
| `fact` | what is true about a person, workspace or project | "churn means cancelled within 30 days"; "the owner wants simple answers" | project notes only |
| `episode` | what happened | "run 12 failed at the clean stage" | experience log and ledger (read through, never copied) |
| `know_how` | how to work | "on HyperAck, `final_customer_fare` is known only after the trip" | lessons; the playbook of Phase 9 (H2, built); skills (H4) planned |
| `knowledge` | rules and evidence for everyone | pitfalls, leakage precedents, experiments | the evidence index (read through, never copied) |

### Five scopes

`global` (DCLab's own: rules, evidence) → `workspace` → `person` → `project` → `session`. A narrower scope wins over a
wider one when two items disagree, and a person's correction wins over an agent's item at any scope.

### One item: `MemoryItem` v1

| Field | Rule |
|---|---|
| `id`, `kind`, `scope`, `scope_id` | required |
| `text` | one line, at most 300 characters (the rule `studio/memory.py` already keeps against injected instructions) |
| `source` | required: the event, run, note or lesson it came from; no source, no item |
| `who` | the person or the agent role that wrote it |
| `status` | `draft` → `active` → `retired` (with the item that replaced it, if any) |
| `supersedes` | the item it corrects |
| `counters` | shown, cited, harmful (moved by code only) |
| `created_at`, `last_used_at` | times |
| `vector` | optional, with the embedding model's id; recomputed when the model changes |

Never in an item: a table's cell value, a key, a password, a sealed set's content, or a value of the target (the same
rules as project notes and the ledger's states).

### Two ways to read

1. **The brief** (always shown): the person's and workspace's active facts, the project's latest notes, the accepted
   know-how of this problem kind, under a token budget (proposal: 800 tokens). It goes into the **first user message**,
   beside today's `memory.summary`, never into the system prompt, so the prompt hashes and the harness gate stay
   meaningful.
2. **`recall(question, scope)`** (on demand): a tool registered once in `agents.default_registry()`, also served on
   `/mcp` for DCLab Code. It filters by scope and kind first, then searches the text (keywords, or keywords plus
   vectors if 35.0 says so). Episodes and knowledge are searched where they already live; nothing is copied.

### Three ways to write

1. **Automatic**, for facts with a source: a project's saved solution, a person's stated preference ("answer in simple
   words"), a correction. The project notes keep being written where they are today.
2. **Reviewed**, for know-how: drafted by an agent, accepted, edited or rejected by a person through `lessons.review`'s
   flow, exactly as lessons are.
3. **Never**: anything that fails the key-name filter (an item naming a column, value or identifier of a benchmark's or
   a challenge's schema: a lesson "group k-fold on order_id" carries the answer to the next run).

### Forgetting

A correction retires what it corrects; near-duplicates merge; an item never cited after a number of briefs (proposal:
30) or marked harmful twice is retired; conflicts that code cannot settle go to a person. Retired is not deleted: the
store keeps everything (the owner's rule of 2026-10-20), and only the owner's audited `store forget` removes an item.

## Rules that bind every package

- The house discipline: failing test first, `make check-quick`, the `dclab-reviewer`, the dev-trajectory record and
  `make sft-dev` in the same commit; `make check-all` after the storage package (35.2).
- Contract and rules in `dclab_core` (no `dclab_rnd` import), wiring in `dclab_rnd`, an adapter and a
  `CORE_INTEGRATION.md` section for the main product; `make core-wheel-check` when the core changes.
- Capture: every write, retirement, brief and recall is an experience event (`memory.written`, `memory.retired`,
  `memory.shown`, `memory.recalled`, `memory.cited`) with its run id, kept whole.
- **Evaluations start with an empty memory**: agent-eval, journeys, the benchmark, the harness suite and every sealed
  read run with memory off unless the arm under test turns it on. A test fails if one of them reads a stored item.
- Consent: person and workspace facts leave the workspace only through `consent.export_*`, redacted, as events do.
- A change to the brief's template is a prompt change: its hash joins `prompts.fingerprints()` and needs
  `make agent-eval` and `make harness-eval`.
- Every embedding request goes through the gateway's `embedding` purpose, local tier; nothing leaves the machine to
  embed a person's words.

## Packages, in order

| # | Package | Needs | Owner action first |
|---|---|---|---|
| 1 | 35.0 Measure embeddings first | — | pull 2–3 embedding models into Ollama |
| 2 | 35.1 The contract | — | — |
| 3 | 35.2 The store | 35.1 | install pgvector only if 35.0 passes |
| 4 | 35.3 Writing | 35.2 | — |
| 5 | 35.4 Reading: the brief and `recall` | 35.3 | — |
| 6 | 35.5 Vectors | 35.0 passed, 35.4 | — |
| 7 | 35.6 Forgetting and upkeep | 35.4 | — |
| 8 | 35.7 The memory page | 35.4 | — |
| 9 | 35.8 Does memory help? | 35.4 (35.5 if built) | budget for live journeys |

### 35.0 Measure embeddings first

- **Delivers:** a memory question set beside `retrieval/questions_v1.json`: about 60 questions over memory-shaped
  items (facts, notes, lessons), at least a third of them reworded so they share no word with their answer, some in
  Persian; split dev and sealed (the sealed part written by an independent agent, hash pinned, read once). The
  retrieval harness measures BM25, hybrid, and each local embedding model (candidates: `qwen3-embedding`, `bge-m3` for
  Persian, `nomic-embed-text`) alone and fused with BM25; recall at 5 with intervals, milliseconds per item, memory
  used on the 8 GB Air.
- **Gate (fixed before the run):** an embedding enters only if it raises recall on the reworded group by more than its
  interval and loses on no other group; otherwise memory stays on keywords, and the result is reported as it is.
  As frozen (`memory_retrieval_gate_v2`, after the review): against `hybrid` (BM25 with LSA, the best search without a
  model; BM25 alone scores 0 on reworded and Persian questions by construction), on the sealed set: reworded mean gain
  at least 0.30 with the 90% bootstrap interval's lower end above 0; direct and Persian each not worse; median query time
  at most 1000 ms. The method is chosen on dev first (the first dev result per candidate). `embeddinggemma` replaced
  `nomic-embed-text`, which does not claim Persian.
- **Done when:** the campaign `memory_retrieval_v1` is stored with its verdict; 35.5 is built or dropped by it.
- **Status (2026-10-28): built, not run.** `dclab_rnd/retrieval/memory.py` and `memory_v1/` (116 invented items, 36 dev
  questions, 30 sealed questions written by an independent agent, hash pinned); the gate frozen in
  `evidence/campaigns/memory_retrieval_v1/manifest.json`; `make memory-retrieval ARGS=…`. Candidates (sizes from
  ollama.com): `qwen3-embedding:0.6b` 639 MB, `bge-m3:567m` 1.2 GB, `embeddinggemma:300m-qat-q8_0` 338 MB. The run
  waits for the owner to pull them (`make memory-retrieval ARGS=candidates` prints the commands) with at least 3 GB of
  disk free. Found while building: keyword search listed the first records by id for a question with no readable word
  (Persian), which counted as found by luck; `retrieval.ranked("bm25")` now drops score-0 hits.

### 35.1 The contract

- **Delivers:** `dclab_core/memory.py`: `MemoryItem` v1 and `Brief` v1 (pydantic, JSON Schema by `make core-schemas`);
  `check(item, schema_names)` (one line, length, a source, no name of a protected schema); `resolve(items)` (scope order,
  corrections first); `select(items, budget)` (the brief under a token budget, deterministic order). The key-name filter
  lives here once; Phase 9's H2 reflector reuses it instead of writing its own.
- **Done when:** an item without a source, with two lines, or naming a protected column is refused; a correction beats
  the item it corrects; a narrower scope beats a wider one; the same items give the same brief; the wheel check passes.
- **Correction (2026-10-28):** this plan was wrong that H2 was only planned: the playbook (`dclab_core/playbook.py`) and
  its screen (`dclab_rnd/learning/filter.py` on `challenge_pack.names_in`) were already built. `challenge_pack` is a
  full-edition module, so `memory.named` is a general copy of the same matching, and a test keeps the two in agreement.
  Playbook bullets stay where they are; 35.2 reads accepted bullets as `know_how` items, as it reads lessons.
- **Status (2026-10-28): built and reviewed** (`dclab_core/memory.py`, dclab-core 0.124.0; `adapters/main_product_memory.py`;
  tests in `test_core_memory.py`, open edition, and `test_core_memory_host.py`). The review's nine findings are fixed,
  among them: a second line hidden with a Unicode separator; ids that merged "< 6" with "> 6"; an agent hiding a person's
  correction; plural column names passing the screen. Open: look-alike letters of another script are not caught, and
  the challenge packs' own `names_in` still misses plurals.

### 35.2 The store

- **Delivers:** `dclab_rnd/memory/store.py`: a `MemoryStore` protocol with a file store (`<workspace>/memory/items.jsonl`,
  append-only) and a PostgreSQL store (`memory_item`, migration 0022, matching `storage/models.py`). Readers (not copies)
  turn project notes and accepted lessons into `MemoryItem`s, so each keeps one source of truth. `migrate` and
  `make backup` cover the table; the store's keepers archive it.
- **Done when:** the same calls pass on files and on PostgreSQL; a project note appears as a `fact` item without being
  written twice; deleting a project retires its items in the product view and keeps them in the store.

### 35.3 Writing

- **Delivers:** `remember(item)` behind `check`: automatic for sourced facts, drafts for know-how (into the lessons
  review flow); a person's correction (`POST /api/memory/{id}/correct`) supersedes; a person's preference said in a
  chat ("answer simply") is proposed as a `person` fact for the person to confirm with one click (never stored silently).
  Every write is an event.
- **Done when:** a planted lesson naming a benchmark's column is refused; a know-how draft is not shown before review; a
  correction retires the old item; text with a line break or an instruction-looking second line is cut to one line.

### 35.4 Reading: the brief and `recall`

- **Delivers:** `brief(scope)` built by `select`, joined to the first user message of the intern, the Home agent, the
  team and the orchestrator (one helper, beside `memory.summary`); `recall` registered once (scopes project, draft,
  session) and on `/mcp`; the run record keeps which item ids were shown and which the reply cited; evaluations run
  with memory off.
- **Done when:** the system prompt's hash is unchanged with and without memory; the brief stays under budget; `recall`
  is a recorded tool call; a test fails if agent-eval, journeys, the benchmark or the harness suite reads a stored item;
  DCLab Code can call `recall` and `remember` through `/mcp` (its panel is that repository's own change).

### 35.5 Vectors (only if 35.0 passed)

- **Delivers:** the vector on each item, computed through the gateway's `embedding` purpose routed to the chosen local
  model; PostgreSQL with pgvector, files with a vector file in the workspace (as `retrieval/` already does); `recall`
  fuses keywords and vectors by reciprocal rank, the way 35.0 measured; recomputed when the model id changes.
- **Done when:** `recall` reproduces 35.0's dev numbers; an item embedded by an old model is recomputed, never mixed.

### 35.6 Forgetting and upkeep

- **Delivers:** `make memory-upkeep`: merge near-duplicates, retire the never-cited and the twice-harmful, list conflicts
  for a person; every change an event; nothing deleted.
- **Done when:** on a planted store the duplicates merge, the stale item retires, a conflict is listed and not settled
  by code, and the store still holds every original.

### 35.7 The memory page

- **Delivers:** a Memory tab in the workspace (web app, `DC.client`, `DC.states`): what is remembered by scope and kind,
  where each item came from, edit, correct, retire; the owner sees every scope, a person their own.
- **Done when:** `make browser-test` covers it at 1280 and 375 px; an edit made there is the next brief's text.

### 35.8 Does memory help?

- **Delivers:** paired dev journeys, memory on against off, the same people and the same order (a second session on a
  project after the first), at least 3 repeats; measured: questions asked again, decisions repeated against a
  correction, tokens, cost; the gate fixed before the run; then one sealed read.
- **Done when:** the result is stored with its intervals. If memory does not help, the brief is cut back to project
  notes and the result is reported as such.

## Decisions the owner makes

| Decision | Proposal |
|---|---|
| Embedding candidates to pull into Ollama | `qwen3-embedding` (small), `bge-m3` (Persian), `nomic-embed-text` |
| pgvector | install with Homebrew only if 35.0 passes; the cloud databases need it too then |
| Person facts | proposed by the agent, confirmed by the person with one click |
| Who reviews know-how | the owner, in the lessons flow |
| Brief budget | 800 tokens |
| Retire after | 30 briefs without a citation, or two harmful marks |

## Risks

- **Memory carries the answer** into a later run or an evaluation: the key-name filter and memory-off evaluations.
- **Stale or wrong memory** steers every later session: sources on every item, corrections first, the memory page.
- **Planted instructions** in a person's text: one-line items, shown as quoted data, never in the system prompt.
- **Cost of the brief** on every turn: a fixed budget; the brief sits in the cached part of the conversation.
- **Vectors chosen by taste**: 35.0's gate decides, and keywords stay if embeddings do not win.
- **Two memories drift apart** (R&D and DCLab Code): DCLab Code reads and writes only through `/mcp`, never a store of
  its own.
