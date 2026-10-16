.PHONY: lake twin-eval tester-setup share collect catalog datasets decisions-gen decision-models decisions-export decision-gate workflow-models solver-data own-models benchmark benchmark-v3 embeddings staging-check model-paths deploy-check browser-test help dev sessions session session-search lineage replay db-reset test-db up down logs scan-secrets sft-v4 test-serial rd-gates retrieval-eval agent-eval agent-eval-live test rd-sync rd-check rd-status rd-baseline rd-smoke rd-campaign-plan rd-campaign-status rd-campaign-report rd-campaign-verify rd-campaign-quick rd-campaign-review agent-serve notebook chat-ui chat-ui-intern mcp-serve agent-test agent-status agent-archive agent-export-clean churn-run churn-status agent-hyperack agent-churn master-guide master-review sft-build report-pdf knowledge index sft-v3 critic-gate pitfalls category-codes copilot-demo product-demo web product-e2e verify-auditor expansion expansion-status new-track research-index clean test-pg check-all check-quick check-quick-tests check-commit core-schemas core-wheel-check code-check journeys harness-eval decision-r3 benchmark-v4 pitfalls-by-kind results-by-kind

PYTHON ?= .venv/bin/python
AGENT_PYTHON ?= .venv-agent/bin/python
# The tests make temporary folders (tempfile.mkdtemp) and do not remove them: hundreds per run of check-all, which
# filled the disk in a day of work. Each test command gets its own TMPDIR, removed when it ends, whatever its result.
TEST_TMP = T=$$(mktemp -d "$${TMPDIR:-/tmp}/dclab-tests.XXXXXX"); trap 'rm -rf "$$T"' EXIT; TMPDIR=$$T
# The suite runs in parallel by default: one process per test file, several at a time, each with its own TMPDIR and its
# own PostgreSQL test database (tests/run_parallel.py). It picks how many from the cores and the memory (6 on an 8 GB M1:
# 98 s, against 205 s for the old single process run and 268 s with 8 jobs, which swap). JOBS=4 sets it; JOBS=1 runs one
# file at a time; `make test-serial` is the old single process run.
JOBS ?=
RUN_TESTS = $(PYTHON) tests/run_parallel.py $(if $(JOBS),--jobs $(JOBS),)
# rd-check's record, freshness and build checks: independent of each other, so they run at the same time
RD_GATES = "-m dclab_rnd validate" "-m dclab_rnd sync --check" "-m dclab_rnd campaign verify" "-m dclab_rnd.evidence_index check" \
	"research/llm-fine-tuning/experiments/sft/build_sft_dataset_v3.py --check" "-m dclab_rnd.research_map --check" \
	"docs/product-demo/build.py --check" "-m dclab_rnd.agentic.web.build --check" "-m dclab_rnd.agent_eval.judged --check" "-m dclab_rnd.data_catalog --check"

help:  ## list the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Automation control plane -------------------------------------------------

test:  ## the whole suite, in parallel (JOBS=N files at a time; JOBS=1 one at a time)
	$(TEST_TMP) $(RUN_TESTS)

test-serial:  ## the whole suite in one process, the way it ran before the parallel runner
	$(TEST_TMP) $(PYTHON) -m unittest discover -s tests -v

rd-sync:  ## rebuild registry + knowledge base
	$(PYTHON) -m dclab_rnd sync

rd-check:  ## record validation, stale-knowledge, campaign and build gates (at the same time, first), then the suite in parallel
	$(MAKE) rd-gates
	$(TEST_TMP) $(RUN_TESTS)

rd-gates:  ## rd-check's checks other than the tests, run at the same time; prints the output of any that fails
	@out=$$(mktemp -d); i=0; for c in $(RD_GATES); do i=$$((i+1)); ( $(PYTHON) $$c > $$out/$$i.log 2>&1; echo $$? > $$out/$$i.rc ) & done; wait; \
	fail=0; i=0; for c in $(RD_GATES); do i=$$((i+1)); if [ "$$(cat $$out/$$i.rc)" = 0 ]; then echo "ok    $$c"; else echo "FAIL  $$c"; cat $$out/$$i.log; fail=1; fi; done; \
	rm -rf $$out; exit $$fail

rd-status:  ## deployment-eligible champions
	$(PYTHON) -m dclab_rnd status

rd-baseline:  ## approve the current champions as the regression baseline
	$(PYTHON) -m dclab_rnd baseline

rd-smoke:  ## fast safe HyperAck experiment through one lifecycle command
	$(PYTHON) -m dclab_rnd cycle hyperack --model logistic_regression --mode safe --optimization baseline


# --- Evidence-first 50-experiment campaign -------------------------------------

rd-campaign-plan:  ## materialize the exact 50-task plan
	$(PYTHON) -m dclab_rnd campaign plan

rd-campaign-status:  ## resumable campaign progress
	$(PYTHON) -m dclab_rnd campaign status

rd-campaign-report:  ## rebuild the campaign report and agent memory
	$(PYTHON) -m dclab_rnd campaign report

rd-campaign-verify:  ## verify campaign evidence and provenance
	$(PYTHON) -m dclab_rnd campaign verify

rd-campaign-quick:  ## run the campaign in quick mode (resumable)
	$(PYTHON) -m dclab_rnd campaign run --quick

rd-campaign-review:  ## run pending LLM critic reviews (needs OPENAI_API_KEY)
	$(PYTHON) -m dclab_rnd campaign review --model gpt-5.6-terra


# --- Agentic Research Studio (agent env, Python 3.12/3.13) ---------------------

agent-serve:  ## the web UI and the API from the agent environment (research campaigns run from the command line: python -m dclab_rnd.agentic run|resume)
	$(AGENT_PYTHON) -m dclab_rnd.agentic serve

up:  ## the product in containers: the API, one job worker and PostgreSQL (docker-compose.yml), on http://127.0.0.1:8765
	docker compose up -d --build --wait

down:  ## stop the containers (the database and the workspace stay in their volumes)
	docker compose down

logs:  ## follow the containers' logs
	docker compose logs -f --tail=100

scan-secrets:  ## look for committed secrets: the tracked files and every line in the history (reports where and what kind, never the value)
	$(PYTHON) scripts/scan_secrets.py
	$(PYTHON) scripts/scan_secrets.py --history

dev:  ## local development on PostgreSQL: checks the server, creates and migrates dclab_dev, runs the API with reload and rebuilds the frontend on change
	$(PYTHON) scripts/dev.py serve

sessions:  ## the latest visits in the development workspace's record (19.2), as make dev's database: ARGS="--limit 50" or "--json"
	DCLAB_DATABASE_URL=$${DCLAB_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_dev} $(PYTHON) -m dclab_rnd.experience sessions $(ARGS)

session:  ## one session's timeline: VISIT=v… (from make sessions), or DRAFT=<draft id>, or PROJECT=<project id>
	DCLAB_DATABASE_URL=$${DCLAB_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_dev} $(PYTHON) -m dclab_rnd.experience show $(VISIT) $(if $(DRAFT),--draft $(DRAFT)) $(if $(PROJECT),--project $(PROJECT)) $(ARGS)

session-search:  ## find events and their sessions (19.4): ARGS="--kind refused", "--project ID", "--text words", "--code DCLAB-R17", "--since 2026-10-13"
	DCLAB_DATABASE_URL=$${DCLAB_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_dev} $(PYTHON) -m dclab_rnd.experience search $(ARGS)

lineage:  ## where a project's numbers come from (19.5): PROJECT=id, PATH_=final.evidence.holdout_metrics.roc_auc for one number
	DCLAB_DATABASE_URL=$${DCLAB_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_dev} $(PYTHON) -m dclab_rnd.experience lineage $(PROJECT) $(if $(PATH_),--path $(PATH_))

replay:  ## rerun a stage from its run record in a scratch copy and compare its number (19.5): PROJECT=id STAGE=final
	DCLAB_DATABASE_URL=$${DCLAB_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_dev} $(PYTHON) -m dclab_rnd.experience replay $(PROJECT) $(STAGE)

db-reset:  ## a clean development database (drops and recreates dclab_dev; needs CONFIRM=yes)
	$(PYTHON) scripts/dev.py db-reset $(if $(filter yes,$(CONFIRM)),--yes,)

backup:  ## the database (pg_dump) and the workspace folder into backups/<time>/, with a manifest of counts and hashes (TO=DIR)
	$(PYTHON) -m dclab_rnd.storage backup --to $(or $(TO),backups)

restore:  ## a backup into a new, empty database (DCLAB_DATABASE_URL) and folder: make restore FROM=backups/<time> HOME_DIR=DIR
	$(PYTHON) -m dclab_rnd.storage restore $(FROM) --home $(HOME_DIR)

test-db:  ## create and migrate the database the tests use (dclab_test)
	$(PYTHON) scripts/dev.py test-db

notebook:  ## start the DCLab notebook UI on http://127.0.0.1:8765 (ML environment; no LLM needed)
	$(PYTHON) -m uvicorn dclab_rnd.agentic.server:app --host 127.0.0.1 --port $${DCLAB_PORT:-8765}

chat-ui:  ## run Hugging Face Chat UI locally with the DCLab notebook as an MCP server (needs Node 20+)
	$(PYTHON) scripts/chat_ui.py

chat-ui-intern:  ## the same with ML Intern mode compiled in (needs a Hugging Face OAuth app for Jobs)
	$(PYTHON) scripts/chat_ui.py --ml-intern

mcp-serve:  ## the DCLab tools as a standalone MCP server on http://127.0.0.1:8777/mcp
	$(PYTHON) -m dclab_rnd.mcp_server

agent-test:  ## agentic tests (agent env) + deterministic worker tests (ML env)
	$(TEST_TMP) $(AGENT_PYTHON) -m pytest tests/test_agentic.py tests/test_research_map.py tests/test_studio.py -q
	$(TEST_TMP) $(PYTHON) -m unittest discover -s tests -p 'test_agentic_worker.py' -v

agent-status:  ## list Studio runs
	$(AGENT_PYTHON) -m dclab_rnd.agentic status

agent-archive:  ## archive every Studio run
	$(AGENT_PYTHON) -m dclab_rnd.agentic archive --all

agent-export-clean:  ## export cleaned Studio traces for SFT
	$(AGENT_PYTHON) -m dclab_rnd.agentic export-clean

churn-run:  ## run the fixed 15-experiment Telco churn campaign
	$(PYTHON) -m dclab_rnd.churn_suite run

churn-status:  ## churn campaign status
	$(PYTHON) -m dclab_rnd.churn_suite status

agent-hyperack:  ## live agentic follow-up research on HyperAck (uses OpenAI)
	$(AGENT_PYTHON) -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4 --rows 11118

agent-churn:  ## live agentic follow-up research on Telco churn (uses OpenAI)
	$(AGENT_PYTHON) -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4 --rows 7043


# --- Knowledge synthesis, SFT corpus, reports ----------------------------------

master-guide:  ## rebuild the model-building field guide and evidence pack
	$(PYTHON) -m dclab_rnd.master_review build

master-review: master-guide  ## typed advisory reviewers, then rebuild the guide
	$(AGENT_PYTHON) -m dclab_rnd.agentic.guide_review --model gpt-5.6-terra
	$(PYTHON) -m dclab_rnd.master_review build

sft-build:  ## build the SFT corpus into research/llm-fine-tuning/experiments/sft/out/
	$(PYTHON) research/llm-fine-tuning/experiments/sft/build_sft_dataset.py --out-dir research/llm-fine-tuning/experiments/sft/out

report-pdf:  ## regenerate research/tabular-classification/reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf
	$(PYTHON) scripts/generate_pdf.py

# --- Knowledge layer: evidence index, SFT v3, copilot, agent tools ---------------

knowledge: index sft-v3  ## rebuild the evidence index and the SFT v3 corpus after new results

index:  ## rebuild evidence/knowledge/rag/records.jsonl
	$(PYTHON) -m dclab_rnd.evidence_index build

sft-v3:  ## rebuild the self-contained SFT v3 corpus into research/llm-fine-tuning/experiments/sft/out_v3/
	$(PYTHON) research/llm-fine-tuning/experiments/sft/build_sft_dataset_v3.py

critic-gate:  ## show LLM critic challenges that the recorded numbers disprove
	$(PYTHON) -m dclab_rnd.critic_gate

pitfalls:  ## re-measure the six common notebook mistakes (evidence/campaigns/pitfalls_v1)
	$(PYTHON) -m dclab_rnd.pitfalls run --force

category-codes:  ## re-measure one-hot vs numeric category codes on identical training folds (evidence/campaigns/category_codes_v1)
	$(PYTHON) -m dclab_rnd.code_encoding run --force

copilot-demo:  ## review the leaky demo notebook and write docs/copilot_demo.html
	$(PYTHON) -m dclab_rnd.copilot review dclab_rnd/copilot/examples/leaky_bank_marketing.ipynb --html docs/copilot_demo.html > /dev/null

sft-v4:  ## corpus v4 (A6.2): v3 plus opted-in projects' trajectories, whole projects held out; writes nothing below 50 trajectories from 10 projects
	$(PYTHON) -m dclab_rnd.studio.corpus_v4

web:  ## rebuild the product frontend (dclab_rnd/agentic/web/src -> dclab_rnd/agentic/static/app)
	$(PYTHON) -m dclab_rnd.agentic.web.build

test-pg:  ## the whole suite on PostgreSQL, in parallel: needs the local database dclab_test (createdb dclab_test); each worker gets dclab_test_p<N>, which the tests empty
	DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(PYTHON) -m dclab_rnd.storage upgrade
	$(TEST_TMP) DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(RUN_TESTS) --pg

# Tests and the end-to-end flows never reach a live model, whatever .env configures (dclab_rnd/models/gateway.py)
test test-serial rd-check test-pg product-e2e check-all: export DCLAB_NO_LIVE_MODELS = 1
# The owner's .env picks the product's model provider; a test sets the model settings it reads itself (read without override)
test test-serial rd-check test-pg product-e2e check-all check-quick check-quick-tests agent-test browser-test: export DCLAB_LLM_PROVIDER =
test test-serial rd-check test-pg product-e2e check-all check-quick check-quick-tests agent-test browser-test: export ANTHROPIC_API_KEY =
test test-serial rd-check test-pg product-e2e check-all check-quick check-quick-tests agent-test browser-test: export ANTHROPIC_MODEL =
browser-test: export DCLAB_NO_LIVE_MODELS = 1

check-all:  ## the gate closing a milestone: rd-check's gates, the suite on files, then on PostgreSQL, the campaign agent's tests (agent env) and the end-to-end flows; what does not need the suite runs beside it
	@out=$$(mktemp -d); \
	( $(MAKE) rd-gates > $$out/gates.log 2>&1; echo $$? > $$out/gates.rc ) & \
	( { if [ -x $(AGENT_PYTHON) ]; then $(TEST_TMP) $(AGENT_PYTHON) -m pytest tests/test_agentic.py -q; else echo "skipped: the campaign tests need $(AGENT_PYTHON)"; fi; } > $$out/agent.log 2>&1; a=$$?; \
	  $(MAKE) product-e2e > $$out/e2e.log 2>&1; e=$$?; echo $$(( a || e )) > $$out/side.rc ) & \
	$(TEST_TMP) $(RUN_TESTS); t=$$?; \
	$(MAKE) test-pg; g=$$?; \
	wait; echo "--- rd-check's gates"; cat $$out/gates.log; echo "--- the campaign agent's tests"; cat $$out/agent.log; echo "--- the end-to-end flows"; cat $$out/e2e.log; \
	r=$$(( t || g || $$(cat $$out/gates.rc) || $$(cat $$out/side.rc) )); rm -rf $$out; \
	if [ $$r = 0 ]; then echo "check-all: passed"; else echo "check-all: FAILED (files $$t, PostgreSQL $$g; the gates and flows above)"; fi; exit $$r

check-quick:  ## the gate for a package, fast: rd-check's gates and the flows (when product code changed) beside the affected tests on files, then the PostgreSQL pass on the files that check it (all when storage changed); check-all closes each milestone
	@out=$$(mktemp -d); \
	( $(MAKE) rd-gates > $$out/gates.log 2>&1; echo $$? > $$out/gates.rc ) & \
	( if [ "$$($(PYTHON) scripts/affected_tests.py --e2e --base $(or $(BASE),HEAD))" = yes ]; then $(MAKE) product-e2e; else echo "flows: not reached by the change"; fi > $$out/e2e.log 2>&1; echo $$? > $$out/e2e.rc ) & \
	$(MAKE) check-quick-tests; t=$$?; wait; echo "--- rd-check's gates"; cat $$out/gates.log; echo "--- the end-to-end flows"; cat $$out/e2e.log; \
	r=$$(( t || $$(cat $$out/gates.rc) || $$(cat $$out/e2e.rc) )); rm -rf $$out; \
	if [ $$r = 0 ]; then echo "check-quick: passed"; else echo "check-quick: FAILED"; fi; exit $$r

check-commit:  ## a gate on the last commit in its own worktree and its own test databases (dclab_check*, made fresh), so work goes on in this one: GATE=check-quick (default) or check-all; the changes since origin/main (BASE=)
	@dir=$$(cd "$${TMPDIR:-/tmp}" && pwd -P)/dclab-check-$$(git rev-parse --short HEAD); rm -rf "$$dir"; git worktree prune; \
	git worktree add --detach "$$dir" HEAD >/dev/null && cp tests/.timings*.json "$$dir/tests/" 2>/dev/null; \
	for v in .venv .venv-agent; do [ -e "$$v" ] && ln -s "$$PWD/$$v" "$$dir/$$v"; done; \
	for n in "" $$(seq -f _p%g 0 16); do dropdb --if-exists "dclab_check$$n" 2>/dev/null; done; createdb dclab_check 2>/dev/null; \
	( cd "$$dir" && DCLAB_TEST_DATABASE_URL=postgresql+psycopg://$$USER@/dclab_check $(MAKE) $(or $(GATE),check-quick) BASE=$(or $(BASE),origin/main) ); r=$$?; \
	git worktree remove --force "$$dir"; echo "check-commit ($(or $(GATE),check-quick)) on $$(git rev-parse --short HEAD): $$( [ $$r = 0 ] && echo passed || echo FAILED )"; exit $$r

check-quick-tests:  ## check-quick's tests: the affected files, then the PostgreSQL pass
	@files=$$($(PYTHON) scripts/affected_tests.py --base $(or $(BASE),HEAD)); echo "files: $$(echo $$files | wc -w | tr -d ' ') test files ($$( [ "$$files" = ALL ] && echo all || echo affected))"; \
	if [ "$$files" = ALL ]; then $(TEST_TMP) $(RUN_TESTS); else $(TEST_TMP) $(RUN_TESTS) $$files; fi
	@pg=$$($(PYTHON) scripts/affected_tests.py --pg --base $(or $(BASE),HEAD)); echo "PostgreSQL: $$( [ "$$pg" = ALL ] && echo every file || echo "$$(echo $$pg | wc -w | tr -d ' ') files that check it")"; \
	DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(PYTHON) -m dclab_rnd.storage upgrade >/dev/null; \
	if [ "$$pg" = ALL ]; then $(TEST_TMP) DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(RUN_TESTS) --pg; \
	elif [ -n "$$pg" ]; then $(TEST_TMP) DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(RUN_TESTS) --pg $$pg; fi

model-paths:  ## every path that uses a model, once, against a real model on made-up data (8.1): ARGS="--ollama qwen2.5-coder:1.5b" (free, local) or ARGS="--configured --cap-eur 2" (costs money)
	$(PYTHON) scripts/model_paths.py $(or $(ARGS),--ollama qwen2.5-coder:1.5b)

deploy-check:  ## the cloud code (deploy/, package 12.6): tofu fmt -check and tofu validate for AWS and Google Cloud, staging and production (downloads the providers once; never plans or applies)
	tofu fmt -check -recursive deploy
	@for d in deploy/aws/staging deploy/aws/production deploy/gcp/staging deploy/gcp/production; do \
	  (cd $$d && TF_PLUGIN_CACHE_DIR=$${TF_PLUGIN_CACHE_DIR:-$$HOME/.terraform.d/plugin-cache} tofu init -backend=false -input=false > /dev/null && tofu validate -no-color) || exit 1; done

browser-test:  ## Chromium drives the real page on a temporary server: Home, the wizard, a run, the brief; 19 pages at 1280 and 375 px (pip install -r requirements/browser.txt)
	$(TEST_TMP) $(PYTHON) scripts/browser_e2e.py

staging-check:  ## the end-to-end flows and the browser checks against a running address, for a staging deployment: make staging-check URL=https://...
	@test -n "$(URL)" || { echo "give the address: make staging-check URL=https://staging.example.com"; exit 2; }
	$(TEST_TMP) $(PYTHON) scripts/product_e2e.py --base $(URL)
	$(TEST_TMP) $(PYTHON) scripts/browser_e2e.py --base $(URL)

product-e2e:  ## run the product's main flows end to end on a temporary server (upload, no data, synthetic, log file; ~3 min, no model)
	$(TEST_TMP) $(PYTHON) scripts/product_e2e.py

product-demo:  ## rebuild docs/product-demo/index.html, the clickable demo of the final product (REFRESH=1 re-extracts the evidence)
	$(PYTHON) docs/product-demo/build.py $(if $(REFRESH),--refresh,)

embeddings:  ## build the evidence record vectors the product searches by (a few thousand tokens; needs a model for the embedding purpose); rebuild after make knowledge
	$(PYTHON) -m dclab_rnd.retrieval --build-embeddings

retrieval-eval:  ## evidence search on the 60-question set (A5.1): recall at 5 for keyword, vector and hybrid; stores a new RET result (ARGS=--embeddings adds the neural retrievers: a fraction of a cent)
	@n=$$(ls evidence/campaigns/retrieval_v1/results/RET-*.json 2>/dev/null | wc -l | tr -d ' '); \
	$(PYTHON) -m dclab_rnd.retrieval $(ARGS) --output evidence/campaigns/retrieval_v1/results/RET-$$(printf '%03d' $$((n + 1)))_recall_at_5.json

dev-archive:  ## copy this project's Claude Code session transcripts into dev_trajectories/raw/ (gitignored, local only: they may hold private data); the curated records beside it are what is committed
	@mkdir -p research/llm-fine-tuning/experiments/dev_trajectories/raw
	rsync -a "$$HOME/.claude/projects/$$(pwd | sed 's#[/ &]#-#g')/" research/llm-fine-tuning/experiments/dev_trajectories/raw/
	@du -sh research/llm-fine-tuning/experiments/dev_trajectories/raw

decisions-gen:  ## 22.1: generated decision benchmarks: invented tables with planted columns, the rules' decisions and their labels, a sealed partition
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.decisions_gen $(ARGS)

decision-models:  ## 22.2: a classifier per decision point on the generated sets (make decisions-gen first), its card beside the rule's dev score
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.decision_models_build $(ARGS)

decisions-export:  ## 22.5: the decision sets as chat JSONL (with a dataset card), a parquet table and typed questions; sealed rows never written
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.decision_export

workflow-models:  ## 26.3: the workflow points' classifiers from journeys: ARGS=collect|train ("gate --yes" judges them once on the sealed journeys)
	DCLAB_NO_LIVE_MODELS=1 DCLAB_DECISION_SHADOW=0 $(PYTHON) -m dclab_rnd.workflow_models $(ARGS)

own-models:  ## 26.7: this workspace's own decision models: ARGS="export|fit|status" (and "gate --yes", once); its rows never leave the workspace folder
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.own_models $(ARGS)

decision-gate:  ## 22.3: each classifier judged once on its sealed decisions by decision_gate_v1 (fixed before the run); prints the plan, ARGS=--yes runs it
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.decision_gate_run $(ARGS)

datasets:  ## 19.12: every training and evaluation set each workspace's consent allows, under <workspace>/datasets/ with their manifests
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.datasets $(ARGS)

sft-dev:  ## the development-trajectory SFT examples (how DCLab is built and fixed) from dev_trajectories/records -> research/llm-fine-tuning/experiments/sft/out_dev/
	$(PYTHON) -m dclab_rnd.devlog

roles-judged:  ## the agentic layer judged (18.9): every role and the team from the result its gate was judged on -> dclab_rnd/roles_judged.json (which roles are on by default)
	$(PYTHON) -m dclab_rnd.agent_eval.judged

kinds-check:  ## every problem kind's acceptance on its replica (16.2): writes dclab_rnd/kinds/capability.json, which the capability matrix reads (no model is called)
	$(TEST_TMP) DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.kinds $(ARGS)

agent-eval:  ## the scripted judgment suite (A4.1): planted traps, the standard plan and two reference policies; writes a new AEV result
	@n=$$(ls evidence/campaigns/agent_eval_v1/results/AEV-*.json 2>/dev/null | wc -l | tr -d ' '); \
	$(TEST_TMP) DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval --output evidence/campaigns/agent_eval_v1/results/AEV-$$(printf '%03d' $$((n + 1)))_judgment_v1_scripted.json

benchmark-v3:  ## benchmark v3 (15.9): v2 reseeded plus rare-event bases and traps; SPLIT=dev (default) or SPLIT=test (sealed: always recorded as a new BEN3 result)
	@n=$$(ls evidence/campaigns/benchmark_v3/results/BEN3-*.json 2>/dev/null | wc -l | tr -d ' '); \
	$(TEST_TMP) DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval.benchmark_v3 --split $(or $(SPLIT),dev) \
	  --output evidence/campaigns/benchmark_v3/results/BEN3-$$(printf '%03d' $$((n + 1)))_$(or $(SPLIT),dev)_scripted.json

benchmark:  ## the frozen benchmark (14.3): scripted policies on SPLIT=dev (default) or SPLIT=test (the sealed set; always recorded as a new BEN result)
	@n=$$(ls evidence/campaigns/benchmark_v2/results/BEN-*.json 2>/dev/null | wc -l | tr -d ' '); \
	$(TEST_TMP) DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval.benchmark --split $(or $(SPLIT),dev) \
	  --output evidence/campaigns/benchmark_v2/results/BEN-$$(printf '%03d' $$((n + 1)))_$(or $(SPLIT),dev)_scripted.json

agent-eval-live:  ## the judgment suite with the configured model (A4.2): prints the plan; ARGS="--yes --cap-eur 2" runs it (caps, repeats, intervals)
	$(PYTHON) -m dclab_rnd.agent_eval.live $(ARGS)

verify-auditor:  ## blind replay of the leakage auditor on datasets with known leaks
	$(PYTHON) -m dclab_rnd.tools verify-auditor

expansion:  ## run the task-type expansion campaign (downloads public data to data/downloads/)
	$(PYTHON) -m dclab_rnd.expansion run

expansion-status:  ## expansion campaign progress
	$(PYTHON) -m dclab_rnd.expansion status

# --- Research tracks -------------------------------------------------------------

new-track:  ## scaffold research/NAME from the template: make new-track NAME=x TITLE="X" PREFIX=X
	@test -n "$(NAME)" || (echo 'usage: make new-track NAME=my-idea TITLE="My idea" PREFIX=MI' && exit 1)
	$(PYTHON) scripts/new_research_track.py "$(NAME)" "$(or $(TITLE),$(NAME))" $(if $(PREFIX),--prefix $(PREFIX),)
	$(PYTHON) -m dclab_rnd.research_map

catalog:  ## regenerate docs/DATA_CATALOG.md and docs/data_catalog.json from the contracts and the code (23.9; rd-gates fails when they drift)
	$(PYTHON) -m dclab_rnd.data_catalog

core-schemas:  ## regenerate dclab_core/schemas/*.json from the contracts (the library the main product adopts)
	$(PYTHON) -m dclab_core.schemas

core-wheel-check:  ## build dclab_core's wheel, install it in a clean Python 3.12 with only its dependencies (uv) and run a smoke call (~90 s)
	$(PYTHON) scripts/core_wheel_check.py

benchmark-v4:  ## 16.14: the benchmark filed by kind, results per kind with intervals (SPLIT=dev, or SPLIT=test once with OUT=evidence/campaigns/benchmark_v4/results/BEN4-<n>_test_scripted.json)
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval.benchmark_v4 --split $(or $(SPLIT),dev) $(if $(OUT),--output $(OUT),)

pitfalls-by-kind:  ## 16.14: each tabular kind's pitfalls on its replica (a wrong split, a kept leak), three seeds, never overwritten
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.pitfalls_by_kind $(ARGS)

results-by-kind:  ## 16.14: one row per kind: its acceptance, the benchmark by kind, its pitfalls, each with its interval
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval.by_kind

decision-r3:  ## 22.7: an R3-shaped run of the decision classifiers against the rule on the generated dev cases, as the main product reads one (its state, its bands)
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.decision_r3 $(ARGS)

tester-setup:  ## 25.11: a trusted tester's first step (NAME=, EMAIL=, REPO= the owner's private records repository); shows the notice, needs AGREE=yes
	$(PYTHON) -m dclab_rnd.testers setup --name "$(NAME)" --email "$(EMAIL)" --repo "$(REPO)" $(if $(filter yes,$(AGREE)),--agree,)

share:  ## 25.11: a tester sends what changed in their workspace (and the tables they agree to) to the owner's records repository
	$(PYTHON) -m dclab_rnd.testers share $(ARGS)

collect:  ## 25.11: the owner pulls the records repository (RECORDS= its clone) and rebuilds every tester under <workspace>/testers/
	$(PYTHON) -m dclab_rnd.testers collect $(if $(RECORDS),--records "$(RECORDS)",) $(ARGS) && $(PYTHON) -m dclab_rnd.testers list

harness-eval:  ## 21.5: the regression cases (the owner's morning first), accepted session cases (ARGS="--home DIR") and the dev journeys, graded step by step; a change to a prompt, a model or a gate is on only after it passes
	$(if $(findstring --live,$(ARGS)),,DCLAB_NO_LIVE_MODELS=1) $(PYTHON) -m dclab_rnd.agent_eval.regression $(ARGS)

lake:  ## 26.1: the training lake: bronze (inputs and hashes), silver (every source normalised, redacted), gold (one set per decision point and task family, checked); manifest in research/llm-fine-tuning/experiments/lake/
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.lake build $(ARGS)

solver-data:  ## 27.5: the lake rebuilt with its cleaning pass (pure states, personal data, near duplicates, evaluation tables, quality, a card per set), then the training files of both models
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.lake build
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.lake export --target decisions
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.lake export --target solver --weights leakage_judgment=2,split_design=2

twin-eval:  ## 25.8: the blind twin runs on studied samples judged against the planted truth, the matrix stored in evidence/campaigns/twin_v1 (ARGS="--run hyperack,telco_churn --yes --cap-eur 1.5" runs Claude first)
	$(PYTHON) -m dclab_rnd.twin eval $(ARGS)

journeys:  ## 20.10: written people carried from the first sentence to the holdout number (SPLIT=dev|sealed; the sealed set runs once)
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) -m dclab_rnd.agent_eval.journeys --split $(or $(SPLIT),dev) $(ARGS)

code-check:  ## 20.9: every kind the engine runs, exported as code, reproduces its holdout number in a clean Python 3.12 (uv; ARGS="binary streams" for some)
	DCLAB_NO_LIVE_MODELS=1 $(PYTHON) scripts/code_check.py $(ARGS)

research-index:  ## regenerate research/<track>/INDEX.md and the track graph
	$(PYTHON) -m dclab_rnd.research_map

clean:  ## remove caches and empty folders (dry run; `make clean APPLY=1` to remove)
	$(PYTHON) scripts/clean_workspace.py $(if $(APPLY),--apply,)

