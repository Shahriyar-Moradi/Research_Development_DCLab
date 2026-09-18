"""Explicit research contexts, never inferred production-availability guarantees."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "adult": {"categorical": "workclass education marital-status occupation relationship race sex native-country".split(), "blocked": [], "decision": "Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review."},
    "bank_marketing": {"categorical": "job marital education default housing loan contact day_of_week month poutcome".split(), "blocked": ["duration"], "decision": "Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification."},
    "breast_cancer": {"categorical": [], "blocked": [], "decision": "Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation."},
    "heart_disease": {"categorical": "sex cp fbs restecg exang slope ca thal".split(), "blocked": [], "decision": "Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity."},
    "credit_default": {"categorical": [f"X{i}" for i in [2, 3, 4, 6, 7, 8, 9, 10, 11]], "blocked": [], "decision": "End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment."},
    "german_credit": {"categorical": [f"Attribute{i}" for i in [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]], "blocked": [], "decision": "Historical credit application research, assuming recorded applicant fields exist at underwriting time. No fairness or lending-use approval."},
    "mushroom": {"categorical": "ALL", "blocked": [], "decision": "Classification after recorded physical traits are observed; never an edibility/safety recommendation."},
    "spambase": {"categorical": [], "blocked": [], "decision": "Email classification after message receipt. Sender-specific proxies and distribution shift remain risks."},
    "online_shoppers": {"categorical": "Month OperatingSystems Browser Region TrafficType VisitorType Weekend".split(), "blocked": ["PageValues", "Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"], "decision": "Session-start purchase prediction. Exclude completed-session counts, durations, bounce/exit rates and PageValues; session-start availability of remaining metadata is assumed, not verified."},
    "wine_quality": {"categorical": [], "blocked": [], "decision": "Wine quality >=6 after physicochemical tests and before quality rating; wine-type/source subgroup information is missing from this legacy cache."},
}

def catalog(root=ROOT):
    result = []
    for key, policy in DATASETS.items():
        path = root / "external_data" / key / "meta.json"
        if path.exists():
            meta = json.loads(path.read_text())
            result.append({**meta, **policy, "key": key, "available": (path.parent / "X.parquet").exists(), "provenance_warning": "Legacy numeric cache: categorical labels were factorized upstream. Treat curated category codes as nominal; original labels and missing-token semantics need raw-source validation. Public UCI data; verify source license before redistribution."})
    return result
