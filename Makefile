.PHONY: help test rd-sync rd-check rd-status rd-baseline rd-smoke rd-campaign-plan rd-campaign-status rd-campaign-report rd-campaign-verify rd-campaign-quick rd-campaign-review agent-serve agent-test agent-status agent-archive agent-export-clean churn-run churn-status agent-hyperack agent-churn master-guide master-review sft-build report-pdf

PYTHON ?= .venv/bin/python
AGENT_PYTHON ?= .venv-agent/bin/python

help:  ## list the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Automation control plane -------------------------------------------------

test:  ## unit tests for the automation layer (ML env)
	$(PYTHON) -m unittest discover -s tests -v

rd-sync:  ## rebuild registry + knowledge base
	$(PYTHON) -m dclab_rnd sync

rd-check:  ## tests, record validation, stale-knowledge and campaign gates
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m dclab_rnd validate
	$(PYTHON) -m dclab_rnd sync --check
	$(PYTHON) -m dclab_rnd campaign verify

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

agent-serve:  ## start the local Studio UI on http://127.0.0.1:8765
	$(AGENT_PYTHON) -m dclab_rnd.agentic serve

agent-test:  ## agentic tests (agent env) + deterministic worker tests (ML env)
	$(AGENT_PYTHON) -m pytest tests/test_agentic.py -q
	$(PYTHON) -m unittest discover -s tests -p 'test_agentic_worker.py' -v

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

sft-build:  ## build the SFT corpus into sft/out/
	$(PYTHON) sft/build_sft_dataset.py --out-dir sft/out

report-pdf:  ## regenerate docs/reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf
	$(PYTHON) scripts/generate_pdf.py
