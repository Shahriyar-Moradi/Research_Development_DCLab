"""What goes into the open-source tree, `dclabai` (the plan: ~/.claude/plans/async-skipping-firefly.md, 2026-10-21).

Read only by scripts/export_open.py. This folder stays private: it names the private words the scanner looks for.

- INCLUDE: globs over `git ls-files` (an untracked file, a .env, a record or a data folder can never reach the export).
- EXCLUDE: globs that win over INCLUDE: the closed parts (training and capture, research and measurement, the twin lane,
  the research campaign, the main product's adapters, the owner's data and documents, the copied UI kit).
- RENAME: the private package name and the public one (the exporter rewrites imports and strings, and moves the folder).
- OVERLAY: files that exist only in the open tree (README, LICENSE, pyproject, docs, CI, the npm launcher).
- LIMITS: the largest file and the binary suffixes allowed.
"""

PUBLIC_NAME = "dclabai"
RENAME = {"dclab_rnd": PUBLIC_NAME}

INCLUDE = [
    "dclab_rnd/**",
    "dclab_core/**",
    "scripts/product_e2e.py",  # the product's flows, against a running server (--base) or one it starts from a checkout
]

EXCLUDE = [
    "dclab_rnd/agentic/web/src/views/lab.html", "dclab_rnd/agentic/web/src/views/benchmark.html",
    "dclab_rnd/agentic/web/src/views/policy.html",  # the research pages (their routes are the full edition's)
    "dclab_rnd/agentic/web/src/views/blueprint.html", "dclab_rnd/agentic/web/src/features.js",  # the planning page (overlay: an empty registry)
    # caches and generated output (the UI is rebuilt inside the export)
    "**/__pycache__/**", "**/*.pyc", "dclab_rnd/agentic/static/app/**",
    # training and capture machinery
    "dclab_rnd/store/**", "dclab_rnd/lake/**", "dclab_rnd/students*.py", "dclab_rnd/archive.py", "dclab_rnd/devlog.py",
    "dclab_rnd/own_llm.py", "dclab_rnd/own_models.py", "dclab_rnd/workflow_models.py", "dclab_rnd/decision_*.py",
    "dclab_rnd/decisions_gen.py", "dclab_rnd/active_learning.py", "dclab_rnd/curriculum.py", "dclab_rnd/committee.py",
    "dclab_rnd/testers/**", "dclab_rnd/datasets.py",
    "dclab_core/store.py", "dclab_core/store_rules.py", "dclab_core/service.py", "dclab_core/lake.py", "dclab_core/decision_models.py",
    "dclab_core/decision_gate.py", "dclab_core/decision_export.py", "dclab_core/benchgen*.py", "dclab_core/committee.py",
    "dclab_core/own_models.py", "dclab_core/datasets.py", "dclab_core/reasoning.py",
    "dclab_core/image_decisions.py", "dclab_core/catalog.py",
    # research and measurement
    "dclab_rnd/agent_eval/**", "dclab_rnd/research_map.py", "dclab_rnd/registry.py", "dclab_rnd/master_review.py",
    "dclab_rnd/critic_gate.py", "dclab_rnd/notebook_assist*.py", "dclab_rnd/notebook_eval.py", "dclab_rnd/cycle.py",
    "dclab_rnd/churn_suite.py", "dclab_rnd/generate_llm_review_pdf.py", "dclab_rnd/llm_review.py", "dclab_rnd/certify.py",
    "dclab_rnd/roles_judged.json", "dclab_rnd/challenge/**", "dclab_rnd/real_problems.py", "dclab_rnd/real_homework.py",
    "dclab_rnd/business_examples.py", "dclab_rnd/knowledge/**", "dclab_rnd/cli.py", "dclab_rnd/agentic/__main__.py",
    "dclab_rnd/agentic/pages/lab.py", "dclab_rnd/agentic/pages/learn.py", "dclab_rnd/agentic/pages/students.py",
    "dclab_rnd/agentic/pages/team.py", "dclab_rnd/agentic/pages/twin.py", "dclab_rnd/agentic/pages/datasets.py",
    "dclab_rnd/studio/corpus_v4.py", "dclab_rnd/studio/sft.py", "dclab_core/adapters/conformance.py",
    "dclab_rnd/evaluation.py", "dclab_rnd/agentic/pages/evaluation.py", "dclab_rnd/data_catalog.py",
    "dclab_core/README.md", "dclab_core/CHANGELOG.md", "dclab_rnd/agentic/web/src/fonts/README.md",
    "dclab_rnd/agentic/engine.py", "dclab_rnd/agentic/agents.py", "dclab_rnd/agentic/campaign_tools.py",
    "dclab_rnd/agentic/live_eval.py", "dclab_rnd/agentic/guide_review.py", "dclab_rnd/agentic/clean_export.py",
    "dclab_core/business_examples.py", "dclab_core/expert.py", "dclab_core/schemas/baseline/**",
    # the twin lane and the orchestration spike
    "dclab_rnd/twin/**", "dclab_core/twin.py", "dclab_rnd/orchestra/**", "dclab_core/orchestra.py",
    "dclab_rnd/prompts/ai_first_*.md", "dclab_rnd/prompts/store_curator.md", "dclab_rnd/prompts/reasoning_teacher.md",
    # the main product's adapters
    "dclab_core/adapters/main_product*",
    # the owner's data and confidential cases
    "dclab_core/replicas/delivery_fraud.py",
    # connectors that need cloud SDKs (v0.2 as extras)
    "dclab_rnd/connectors/cloud.py", "dclab_rnd/connectors/hf.py",
    # the copied UI kit and baked research records (replaced by the open theme and the shipped knowledge)
    "dclab_rnd/agentic/web/src/kit/**", "dclab_rnd/agentic/web/src/data/records.json", "dclab_rnd/agentic/web/src/data/sft.json",
    "dclab_rnd/agentic/web/src/data/copilot_review.json",
]

# Modules the open tree may still name inside `optional(...)` (a seam): absent there, present here.
CLOSED = [
    "store", "lake", "students", "students_loop", "students_exam", "archive", "devlog", "own_llm", "own_models", "workflow_models",
    "agent_eval", "research_map", "twin", "orchestra", "testers", "business_examples", "real_problems", "committee", "curriculum",
    "datasets", "registry", "master_review", "critic_gate", "notebook_assist", "certify", "cli", "agentic.engine", "agentic.agents",
    "agentic.campaign_tools", "agentic.live_eval", "agentic.guide_review", "agentic.clean_export", "connectors.cloud", "connectors.hf",
]

# Words replaced in every exported text file, in this order (after the package rename): the owner's real dataset and the mock
# user become invented names, private working files become the public ones. The scanner then checks nothing private is left.
REPLACE = [
    (r"make kinds-check ARGS=", "python -m dclabai.kinds "), (r"make kinds-check", "python -m dclabai.kinds"),
    (r"hyper_ackt-dataset\.csv", "ride_fares.csv"),
    (r"hyper_ackt", "ride_fares"),
    (r"hyper_ack\b", "accepted"),
    (r"HYPERACK", "RIDEFARE"), (r"HyperAck", "RideFare"), (r"Hyperack", "Ridefare"), (r"hyperack", "ridefare"),
    (r"shahriar994@gmail\.com", "alex@example.com"),
    (r"Shahriyar Moradi", "Alex Morgan"), (r"Shahriar Moradi", "Alex Morgan"),
    (r"Shahriyar", "Alex"), (r"Shahriar", "Alex"), (r"shahriyar", "alex"), (r"shahriar", "alex"),
    (r"TripsSource design kit", "design kit"), (r"TripsSource's", "the kit's"), (r"TripsSource", "the kit"),
    (r"CLAUDE\.md", "CONTRIBUTING.md"),
    (r"Tripssource_UI/[^\s)]*", "the design kit"), (r"(?i)tripssource", "the kit"),
    (r"--ts-", "--dc-"),
    (r"/Users/[A-Za-z0-9._-]+/[^\s'\"]*", "/home/user/project"),
]

OVERLAY = "export/overlay"  # files only the open tree has (README, LICENSE, pyproject, docs, CI, the npm launcher)

# The tests the open tree ships: every test file whose in-house imports are all exported and that names none of the
# private folders (chosen by scripts/export_open.py, each left-out file with its reason in the report), plus helpers.
TESTS = {
    "candidates": "tests/test_*.py",
    "helpers": ["tests/pgtest.py", "tests/fixtures/**"],
    "private_paths": ["data/project", "data/public", "data/downloads", "evidence/", "research/", "training_records", "agent_runs/",
                      "docs/guides", "docs/product-demo", "/sft/", "decisions/", "dev_trajectories", "make rd-check"],
    "exclude": {  # a test file left out by hand, with why (each found by running the selection in a fresh install)
        "tests/test_config.py": "reads the repository with git and scripts/scan_secrets.py; back when the open repository's CI runs it",
        "tests/test_core_wheel.py": "checks dclab_core's own wheel and changelog, which stay in the private repository",
        "tests/test_core_contracts.py": "checks every contract against the schema baselines, which stay in the private repository",
        "tests/test_dev_script.py": "tests scripts/dev.py (the private repository's development runner)",
        "tests/test_deploy.py": "tests the cloud code in deploy/, which is not exported",
        "tests/test_staging_check.py": "tests a Makefile target of the private repository",
        "tests/test_decision_routing.py": "routes measured classifiers: needs the sealed gate results in evidence/ and the agent_eval suites",
        "tests/test_kind_acceptance_rare.py": "the acceptance of the delivery_fraud replica, which is not part of the open edition yet",
        "tests/test_code.py": "built around the anomaly kind's case, whose replica the open edition does not have yet",
    },
}

# Kinds whose acceptance case (an invented replica) is not exported: the open tree's capability matrix says "partial"
# for them, without the checks (their details describe that replica).
KINDS_WITHOUT_CASE = {"rare_event": "its acceptance replica is not part of the open edition yet",
                      "anomaly": "its acceptance replica is not part of the open edition yet"}

# The knowledge the open tree ships (the owner's decision, 2026-10-21: rules and pitfalls only), filtered from the evidence
# index and packaged where evidence_index.packaged_records() reads it. Citations keep only paths that exist in the open tree.
KNOWLEDGE = {"source": "evidence/knowledge/rag/records.jsonl", "types": ["rule", "pitfall"], "out": "knowledge/records.jsonl"}

MAX_FILE_BYTES = 1_000_000
BINARY_ALLOWED = (".woff2", ".woff", ".ttf", ".png", ".svg", ".ico")

# The private words file (one pattern per line, regular expressions, case-insensitive) is export/forbidden.txt: never exported.
FORBIDDEN_FILE = "export/forbidden.txt"
