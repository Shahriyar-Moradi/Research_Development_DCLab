.PHONY: test rd-sync rd-check rd-status rd-baseline rd-smoke

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m unittest discover -s tests -v

rd-sync:
	$(PYTHON) -m dclab_rnd sync

rd-check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m dclab_rnd validate
	$(PYTHON) -m dclab_rnd sync --check

rd-status:
	$(PYTHON) -m dclab_rnd status

rd-baseline:
	$(PYTHON) -m dclab_rnd baseline

rd-smoke:
	$(PYTHON) -m dclab_rnd cycle hyperack --model logistic_regression --mode safe --optimization baseline
