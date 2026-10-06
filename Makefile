.PHONY: help test rd-sync rd-check rd-status rd-baseline rd-smoke rd-campaign-plan rd-campaign-status rd-campaign-report rd-campaign-verify rd-campaign-quick rd-campaign-review agent-serve notebook chat-ui chat-ui-intern mcp-serve agent-test agent-status agent-archive agent-export-clean churn-run churn-status agent-hyperack agent-churn master-guide master-review sft-build report-pdf knowledge index sft-v3 critic-gate pitfalls category-codes copilot-demo product-demo web product-e2e verify-auditor expansion expansion-status new-track research-index clean test-pg check-all

PYTHON ?= .venv/bin/python
AGENT_PYTHON ?= .venv-agent/bin/python
# The tests make temporary folders (tempfile.mkdtemp) and do not remove them: hundreds per run of check-all, which
# filled the disk in a day of work. Each test command gets its own TMPDIR, removed when it ends, whatever its result.
TEST_TMP = T=$$(mktemp -d "$${TMPDIR:-/tmp}/dclab-tests.XXXXXX"); trap 'rm -rf "$$T"' EXIT; TMPDIR=$$T

help:  ## list the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Automation control plane -------------------------------------------------

test:  ## unit tests for the automation layer (ML env)
	$(TEST_TMP) $(PYTHON) -m unittest discover -s tests -v

rd-sync:  ## rebuild registry + knowledge base
	$(PYTHON) -m dclab_rnd sync

rd-check:  ## tests, record validation, stale-knowledge and campaign gates
	$(TEST_TMP) $(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m dclab_rnd validate
	$(PYTHON) -m dclab_rnd sync --check
	$(PYTHON) -m dclab_rnd campaign verify
	$(PYTHON) -m dclab_rnd.evidence_index check
	$(PYTHON) research/llm-fine-tuning/experiments/sft/build_sft_dataset_v3.py --check
	$(PYTHON) -m dclab_rnd.research_map --check
	$(PYTHON) docs/product-demo/build.py --check
	$(PYTHON) -m dclab_rnd.agentic.web.build --check

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

agent-serve:  ## start the web UI with the LLM campaigns enabled (agent environment)
	$(AGENT_PYTHON) -m dclab_rnd.agentic serve

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

web:  ## rebuild the product frontend (dclab_rnd/agentic/web/src -> dclab_rnd/agentic/static/app)
	$(PYTHON) -m dclab_rnd.agentic.web.build

test-pg:  ## the whole suite on PostgreSQL: needs the local database dclab_test (createdb dclab_test); the tests empty it
	DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(PYTHON) -m dclab_rnd.storage upgrade
	$(TEST_TMP) DCLAB_DATABASE_URL=$${DCLAB_TEST_DATABASE_URL:-postgresql+psycopg://$$USER@/dclab_test} $(PYTHON) -m unittest discover -s tests

# Tests and the end-to-end flows never reach a live model, whatever .env configures (dclab_rnd/models/gateway.py)
rd-check test-pg product-e2e check-all: export DCLAB_NO_LIVE_MODELS = 1

check-all:  ## the gate for every package: rd-check on files, the suite on PostgreSQL, the campaign agent's tests (agent env) and the end-to-end flows
	$(MAKE) rd-check
	$(MAKE) test-pg
	@if [ -x $(AGENT_PYTHON) ]; then $(TEST_TMP) $(AGENT_PYTHON) -m pytest tests/test_agentic.py -q; else echo "skipped: the campaign tests need $(AGENT_PYTHON)"; fi
	$(MAKE) product-e2e

product-e2e:  ## run the product's main flows end to end on a temporary server (upload, no data, synthetic, log file; ~3 min, no model)
	$(TEST_TMP) $(PYTHON) scripts/product_e2e.py

product-demo:  ## rebuild docs/product-demo/index.html, the clickable demo of the final product (REFRESH=1 re-extracts the evidence)
	$(PYTHON) docs/product-demo/build.py $(if $(REFRESH),--refresh,)

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

research-index:  ## regenerate research/<track>/INDEX.md and the track graph
	$(PYTHON) -m dclab_rnd.research_map

clean:  ## remove caches and empty folders (dry run; `make clean APPLY=1` to remove)
	$(PYTHON) scripts/clean_workspace.py $(if $(APPLY),--apply,)

