.PHONY: test rd-sync rd-check rd-status rd-baseline rd-smoke rd-campaign-plan rd-campaign-status rd-campaign-report rd-campaign-verify rd-campaign-quick rd-campaign-review agent-serve agent-test agent-status churn-run churn-status agent-hyperack agent-churn

PYTHON ?= .venv/bin/python
AGENT_PYTHON ?= .venv-agent/bin/python

test:
	$(PYTHON) -m unittest discover -s tests -v

rd-sync:
	$(PYTHON) -m dclab_rnd sync

rd-check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m dclab_rnd validate
	$(PYTHON) -m dclab_rnd sync --check
	$(PYTHON) -m dclab_rnd campaign verify

rd-status:
	$(PYTHON) -m dclab_rnd status

rd-baseline:
	$(PYTHON) -m dclab_rnd baseline

rd-smoke:
	$(PYTHON) -m dclab_rnd cycle hyperack --model logistic_regression --mode safe --optimization baseline

rd-campaign-plan:
	$(PYTHON) -m dclab_rnd campaign plan

rd-campaign-status:
	$(PYTHON) -m dclab_rnd campaign status

rd-campaign-report:
	$(PYTHON) -m dclab_rnd campaign report

rd-campaign-verify:
	$(PYTHON) -m dclab_rnd campaign verify

rd-campaign-quick:
	$(PYTHON) -m dclab_rnd campaign run --quick

rd-campaign-review:
	$(PYTHON) -m dclab_rnd campaign review --model gpt-5.4-mini

agent-serve:
	$(AGENT_PYTHON) -m dclab_rnd.agentic serve

agent-test:
	$(AGENT_PYTHON) -m pytest tests/test_agentic.py -q
	$(PYTHON) -m unittest discover -s tests -p 'test_agentic_worker.py' -v

agent-status:
	$(AGENT_PYTHON) -m dclab_rnd.agentic status

churn-run:
	$(PYTHON) -m dclab_rnd.churn_suite run

churn-status:
	$(PYTHON) -m dclab_rnd.churn_suite status

agent-hyperack:
	$(AGENT_PYTHON) -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4 --rows 11118

agent-churn:
	$(AGENT_PYTHON) -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4 --rows 7043
