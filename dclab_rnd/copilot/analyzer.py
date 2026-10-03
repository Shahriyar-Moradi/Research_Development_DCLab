"""Static, order-aware methodology detectors for notebooks.

Every detector is deliberately conservative: it reports what it can see in the
code and says what it cannot know (static analysis cannot see column names that
only exist in the data). Severity is calibrated by the pitfalls campaign
(``evidence/campaigns/pitfalls_v1``): a mistake whose measured cost is large is ``high``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dclab_rnd.evidence_index import ROOT, EvidenceIndex

SPLIT_FUNCS = {"train_test_split"}
SPLITTER_CLASSES = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit",
                    "ShuffleSplit", "StratifiedShuffleSplit", "RepeatedStratifiedKFold", "RepeatedKFold"}
TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "SimpleImputer", "KNNImputer", "IterativeImputer", "OneHotEncoder", "OrdinalEncoder",
    "TargetEncoder", "LabelEncoder", "PCA", "TruncatedSVD", "TfidfVectorizer", "CountVectorizer",
    "PolynomialFeatures", "KBinsDiscretizer",
}
SELECTORS = {"SelectKBest", "SelectPercentile", "RFE", "RFECV", "SelectFromModel", "VarianceThreshold",
             "SequentialFeatureSelector", "GenericUnivariateSelect"}
RESAMPLERS = {"SMOTE", "ADASYN", "BorderlineSMOTE", "RandomOverSampler", "RandomUnderSampler", "SMOTENC", "SMOTETomek"}
SEARCHERS = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict"}
RICH_METRICS = {"roc_auc_score", "average_precision_score", "f1_score", "log_loss", "brier_score_loss",
                "classification_report", "precision_recall_curve", "roc_curve", "precision_score", "recall_score",
                "balanced_accuracy_score", "matthews_corrcoef", "mean_absolute_error", "mean_squared_error", "r2_score"}
RANDOM_ESTIMATORS = {"RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
                     "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
                     "HistGradientBoostingRegressor", "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
                     "CatBoostClassifier", "CatBoostRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor",
                     "MLPClassifier", "MLPRegressor"}
FAMILIES = {
    "linear": {"LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier", "RidgeClassifier",
               "LinearSVC", "LogisticRegressionCV"},
    "svm": {"SVC", "SVR", "NuSVC"},
    "tree": {"DecisionTreeClassifier", "DecisionTreeRegressor"},
    "bagging": {"RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "BaggingClassifier"},
    "boosting": {"GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
                 "HistGradientBoostingRegressor", "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
                 "CatBoostClassifier", "CatBoostRegressor", "AdaBoostClassifier"},
    "neighbors": {"KNeighborsClassifier", "KNeighborsRegressor"},
    "naive_bayes": {"GaussianNB", "MultinomialNB", "BernoulliNB"},
    "neural": {"MLPClassifier", "MLPRegressor"},
    "baseline": {"DummyClassifier", "DummyRegressor"},
}
ESTIMATORS = set().union(*FAMILIES.values())
POST_OUTCOME_NAME = re.compile(
    r"(^|[_\s])(final|after|post|outcome|result|resolved|resolution|closed|refund(ed)?|cancel(l)?ed|cancellation|"
    r"churn(ed)?_date|end_date|duration|settled|paid_amount|chargeback|returned|label|target_mean)([_\s]|$)",
    re.IGNORECASE,
)
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
KNOWN_DATASET_PATTERNS = {
    "hyperack": r"hyper_?ackt?[-_]dataset|hyperack",
    "telco_churn": r"Telco-Customer-Churn",
    "bank_marketing": r"bank_marketing|bank-full|bank-additional",
    "online_shoppers": r"online_shoppers",
    "credit_card_fraud": r"creditcard\.csv|credit_card_fraud",
    "bike_sharing_daily": r"bike_sharing|bike-sharing|\bbike\.csv",
    "ecommerce_clothing_reviews": r"Clothing E-?Commerce Reviews|ecommerce_clothing_reviews",
}


def blocked_columns() -> dict[str, list[str]]:
    """Decision-time blocked columns per known dataset (from the agentic catalog and expansion adapters)."""
    blocked: dict[str, list[str]] = {}
    try:
        from dclab_rnd.agentic.catalog import DATASETS

        blocked.update({k: list(v.get("blocked", [])) for k, v in DATASETS.items() if v.get("blocked")})
    except Exception:
        pass
    try:
        from dclab_rnd.expansion.datasets import dataset_cards

        for card in dataset_cards():
            if card.get("blocked_features"):
                blocked[card["key"]] = list(card["blocked_features"])
    except Exception:
        pass
    return blocked


@dataclass
class Finding:
    detector: str
    severity: str
    cell: int
    line: int
    title: str
    message: str
    suggestion: str
    rules: list[str]
    precedent_query: str = ""
    precedent_filters: dict[str, Any] = field(default_factory=dict)
    proof: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- notebook parsing


def load_cells(path: Path) -> list[dict[str, Any]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        cells.append({"index": i, "type": cell.get("cell_type", "code"), "source": source})
    return cells


def _clean_source(source: str) -> str:
    """Comment out IPython magics and shell escapes so the cell parses as Python."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        lines.append(("# " + line) if stripped.startswith(("%", "!", "?")) else line)
    return "\n".join(lines)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _root_name(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node.id if isinstance(node, ast.Name) else ""


def _kw(node: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in node.keywords if k.arg == name), None)


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _strings_in(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_train_name(name: str) -> bool:
    return bool(re.search(r"train|tr\b|_tr$|fit", name, re.IGNORECASE))


def _is_test_name(name: str) -> bool:
    return bool(re.search(r"test|holdout|valid|val\b|_te$|eval", name, re.IGNORECASE))


# --------------------------------------------------------------------------- analysis state


@dataclass
class _State:
    split_seen: bool = False
    split_cell: int | None = None
    instances: dict[str, str] = field(default_factory=dict)  # variable -> class name
    estimator_classes: list[tuple[str, int, int]] = field(default_factory=list)
    metrics_used: set[str] = field(default_factory=set)
    accuracy_sites: list[tuple[int, int]] = field(default_factory=list)
    timestamp_hint: tuple[int, int] | None = None
    time_aware_split: bool = False
    random_splits: list[tuple[int, int, ast.Call]] = field(default_factory=list)
    classifier_used: bool = False
    columns: dict[str, tuple[int, int]] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    datasets_read: dict[str, tuple[int, int]] = field(default_factory=dict)


def _record_columns(strings: list[str], cell: int, line: int, state: _State) -> None:
    for value in strings:
        if re.fullmatch(r"[A-Za-z][\w\- ]{0,60}", value) and value not in state.columns:
            state.columns[value] = (cell, line)


def _analyze_cell(tree: ast.AST, cell: int, state: _State, findings: list[Finding]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            cls = _call_name(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    state.instances[target.id] = cls

    # column names: only strings used as column selectors, never arbitrary text
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            _record_columns(_strings_in(node.slice), cell, node.lineno, state)
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            if any(isinstance(t, ast.Name) and re.search(r"feat|col|drop|keep|use", t.id, re.IGNORECASE) for t in node.targets):
                _record_columns(_strings_in(node.value), cell, node.lineno, state)

    # statements in source order (ast.walk is breadth-first; sort by position)
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]

    for call in calls:
        name = _call_name(call)
        line = call.lineno

        # ---- collect facts -------------------------------------------------------
        if name in ESTIMATORS:
            state.estimator_classes.append((name, cell, line))
            if name.endswith("Classifier") or name in {"LogisticRegression", "SVC", "LinearSVC", "GaussianNB"}:
                state.classifier_used = True
        if name in RICH_METRICS:
            state.metrics_used.add(name)
        if name == "accuracy_score":
            state.accuracy_sites.append((cell, line))
        if name in {"to_datetime", "TimeSeriesSplit"} or (name == "read_csv" and _kw(call, "parse_dates")):
            state.timestamp_hint = state.timestamp_hint or (cell, line)
        if name == "TimeSeriesSplit":
            state.time_aware_split = True
        if name == "drop":
            state.dropped.update(_strings_in(call))
        for kw in call.keywords:
            if kw.arg in {"columns", "subset", "labels", "usecols"} and name != "drop":
                _record_columns(_strings_in(kw.value), cell, line, state)
        if any(re.search(r"(^|_)(date|time|timestamp|created_at|datetime)($|_)", s, re.IGNORECASE) for s in _strings_in(call)):
            state.timestamp_hint = state.timestamp_hint or (cell, line)

        # ---- recognise datasets the R&D already knows ------------------------------------
        if name.startswith("read_") or name in {"open", "load"} or "/" in "".join(_strings_in(call)):
            joined = " ".join(_strings_in(call)) + " " + (ast.unparse(call) if hasattr(ast, "unparse") else "")
            for key, pattern in KNOWN_DATASET_PATTERNS.items():
                if re.search(pattern, joined, re.IGNORECASE):
                    state.datasets_read.setdefault(key, (cell, line))

        # ---- absolute paths ---------------------------------------------------------
        if name.startswith("read_") or name in {"open", "load"}:
            for s in _strings_in(call):
                if re.match(r"(/Users/|/home/|[A-Za-z]:\\)", s):
                    findings.append(Finding(
                        "absolute_data_path", "low", cell, line, "Hard-coded absolute data path",
                        f"`{s[:80]}` only exists on one machine, so nobody else can rerun this notebook and the data "
                        "version is not recorded.",
                        "Load from a path relative to the project (e.g. `Path('data') / 'file.csv'`) and record the file's SHA-256 next to results.",
                        ["DCLAB-R21"], "immutable snapshot hash source lineage", {"type": "workflow"},
                    ))

        # ---- split detection --------------------------------------------------------
        if name in SPLIT_FUNCS or (name == "split" and state.instances.get(_root_name(call.func)) in SPLITTER_CLASSES):
            if name in SPLIT_FUNCS:
                state.random_splits.append((cell, line, call))
                if _kw(call, "random_state") is None:
                    findings.append(Finding(
                        "missing_random_state", "low", cell, line, "Split is not reproducible",
                        "`train_test_split` has no `random_state`, so every run evaluates on different rows and results cannot be compared or reproduced.",
                        "Add `random_state=42` (any fixed number) and keep it fixed for every experiment you compare.",
                        ["DCLAB-R16", "DCLAB-R21"], "paired baseline same folds reproducible", {"type": "rule"},
                    ))
                shuffle = _kw(call, "shuffle")
                if shuffle is not None and isinstance(shuffle.value, ast.Constant) and shuffle.value.value is False:
                    state.time_aware_split = True
            if not state.split_seen:
                state.split_seen, state.split_cell = True, cell

        # ---- fit-before-split family -----------------------------------------------
        if name in {"fit", "fit_transform", "fit_resample"} and isinstance(call.func, ast.Attribute):
            receiver = call.func.value
            cls = state.instances.get(_root_name(receiver), "") if isinstance(receiver, ast.Name) else (
                _call_name(receiver) if isinstance(receiver, ast.Call) else "")
            arg_names = set().union(*(_names_in(a) for a in call.args)) if call.args else set()
            on_test = any(_is_test_name(n) for n in arg_names)
            if on_test and name != "fit_resample" and cls not in ESTIMATORS | SEARCHERS:
                findings.append(Finding(
                    "fit_on_test", "high", cell, line, "Transformer fitted on test data",
                    f"`{cls or 'object'}.{name}(...)` receives test data. The test set must only ever be transformed with "
                    "statistics learned from training data, otherwise the evaluation has seen the answers.",
                    "Fit on the training data once (`fit` or `fit_transform` on X_train) and call only `transform` on X_test.",
                    ["DCLAB-R03", "DCLAB-R17"], "fit statistics training partitions only", {"type": "pitfall"},
                ))
            if cls in RESAMPLERS or name == "fit_resample":
                if not state.split_seen:
                    findings.append(Finding(
                        "resample_before_split", "high", cell, line, "Oversampling before the split",
                        f"`{cls or 'resampler'}.{name}` runs before any train/test split. Copies or interpolations of minority rows "
                        "end up in both train and test, so the model is graded on rows it has effectively seen.",
                        "Split first, then resample the training part only; better, put the sampler inside an `imblearn.pipeline.Pipeline` so it runs inside each CV fold.",
                        ["DCLAB-R02", "DCLAB-R04"], "oversampling minority before split inflated", {"type": "pitfall"},
                    ))
                elif on_test:
                    findings.append(Finding(
                        "resample_test", "high", cell, line, "Resampling the test set",
                        "The evaluation data is being resampled. Test data must keep its natural class balance or the metrics describe a world that does not exist.",
                        "Resample training data only and report metrics on untouched test rows (average precision suits imbalance).",
                        ["DCLAB-R18"], "oversampling minority before split inflated", {"type": "pitfall"},
                    ))
            elif cls in SELECTORS and not state.split_seen:
                findings.append(Finding(
                    "selection_before_split", "high", cell, line, "Feature selection sees every label",
                    f"`{cls}.{name}` is fitted before the data is split, so the chosen features were picked using the test labels too.",
                    "Move the selector into a `Pipeline` together with the model and evaluate the whole pipeline with cross-validation.",
                    ["DCLAB-R03", "DCLAB-R02"], "feature selection before cross validation noise", {"type": "pitfall"},
                ))
            elif cls in TRANSFORMERS and not state.split_seen:
                target_aware = cls in {"TargetEncoder"}
                findings.append(Finding(
                    "preprocess_before_split", "high" if target_aware else "medium", cell, line,
                    "Preprocessing fitted before the split",
                    f"`{cls}.{name}` learns statistics from all rows before the train/test split. For simple scaling the measured "
                    "effect is tiny, but it builds the habit that makes selection, resampling and encoding leak badly.",
                    "Create the split first, or wrap preprocessing and model in one `Pipeline` so every statistic is learned from training data only.",
                    ["DCLAB-R03", "DCLAB-R02"], "preprocess imputation scaling before split", {"type": "pitfall"},
                ))
            elif cls in SEARCHERS and not state.split_seen:
                findings.append(Finding(
                    "search_on_all_data", "medium", cell, line, "Hyperparameter search on all rows",
                    f"`{cls}.fit` runs before any holdout was set aside, so the best CV score is both the tuning target and the reported result.",
                    "Hold out a test set first and run the search on training data only; report the test score once at the end.",
                    ["DCLAB-R15", "DCLAB-R17"], "best of many configurations holdout optimistic", {"type": "pitfall"},
                ))

        # ---- function-style oversampling (sklearn.utils.resample, sample(replace=True)) ---
        replace_kw = _kw(call, "replace")
        is_upsample = name == "resample" or (
            name == "sample" and replace_kw is not None and isinstance(replace_kw.value, ast.Constant) and replace_kw.value.value is True
        )
        if is_upsample and not state.split_seen:
            findings.append(Finding(
                "resample_before_split", "high", cell, line, "Oversampling before the split",
                "Rows are duplicated (resampled with replacement) before any train/test split. Copies of the same row end up "
                "in both train and test, so the model is graded on rows it has already memorized.",
                "Split first, then oversample the training part only; better, use `imblearn.pipeline.Pipeline` so it happens inside each CV fold.",
                ["DCLAB-R02", "DCLAB-R04"], "oversampling minority before split inflated", {"type": "pitfall"},
            ))

        # ---- target aggregates before split ------------------------------------------
        if name in {"transform", "agg", "mean"} and not state.split_seen and isinstance(call.func, ast.Attribute):
            chain = ast.unparse(call.func.value) if hasattr(ast, "unparse") else ""
            if "groupby" in chain and (name == "mean" or any(s in {"mean", "sum"} for s in _strings_in(call))):
                aggregated = _strings_in(call.func.value)
                target_like = any(re.fullmatch(r"(target|label|y|class|churn|outcome|is_fraud|default|hyper_ack)", v, re.IGNORECASE) for v in aggregated)
                findings.append(Finding(
                    "aggregate_before_split", "high" if target_like else "medium", cell, line,
                    "Target-mean feature computed on all rows" if target_like else "Group aggregate computed on all rows",
                    "A groupby aggregate is computed before the split. If the aggregated column is (or depends on) the target, "
                    "each row's own label is inside its feature: target-mean encoding leakage.",
                    "Compute group statistics out-of-fold on training data (e.g. sklearn `TargetEncoder`, which cross-fits) and apply them to test rows.",
                    ["DCLAB-R04", "DCLAB-R03"], "target mean encoding before split identifier", {"type": "pitfall"},
                ))
        if name == "fillna" and not state.split_seen:
            if any(_call_name(a) in {"mean", "median", "mode"} for a in call.args if isinstance(a, ast.Call)):
                findings.append(Finding(
                    "impute_before_split", "low", cell, line, "Imputation statistic from all rows",
                    "Missing values are filled with a statistic computed on all rows, including future test rows.",
                    "Use `SimpleImputer` inside a `Pipeline` fitted on training data.",
                    ["DCLAB-R03", "DCLAB-R12"], "preprocess imputation scaling before split", {"type": "pitfall"},
                ))

        # ---- random_state on stochastic estimators ------------------------------------
        if name in RANDOM_ESTIMATORS and _kw(call, "random_state") is None and _kw(call, "seed") is None and _kw(call, "random_seed") is None:
            findings.append(Finding(
                "missing_random_state", "low", cell, line, f"`{name}` is not seeded",
                f"`{name}` is random; without a seed two runs give different scores, so small 'improvements' can be noise.",
                f"Pass `random_state=42` to `{name}` and keep it fixed across compared runs.",
                ["DCLAB-R16"], "paired baseline same folds predeclared margin", {"type": "rule"},
            ))

    # ---- holdout reuse inside loops ----------------------------------------------------
    for loop in loops:
        loop_calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)]
        fits = [c for c in loop_calls if _call_name(c) == "fit"]
        scored_on_test = [
            c for c in loop_calls
            if (_call_name(c) in RICH_METRICS | {"accuracy_score", "score"})
            and any(_is_test_name(n) for a in c.args for n in _names_in(a))
        ]
        if fits and scored_on_test:
            findings.append(Finding(
                "holdout_reuse", "medium", cell, loop.lineno, "Test set used to choose between models",
                "Inside this loop several models are fitted and scored on the test set. Picking the best of them by test score "
                "turns the test set into a tuning set, so its score is now optimistic.",
                "Compare candidates with cross-validation on training data (`cross_val_score` / `GridSearchCV`), then score only the final choice on the test set once.",
                ["DCLAB-R17", "DCLAB-R16"], "best of many configurations holdout optimistic selection", {"type": "pitfall"},
            ))


def _notebook_level(state: _State, findings: list[Finding], known_leaks: dict[str, dict[str, Any]]) -> None:
    families = {fam for cls, _, _ in state.estimator_classes for fam, members in FAMILIES.items() if cls in members}
    if state.estimator_classes and len(families - {"baseline"}) == 1 and "baseline" not in families:
        cls, cell, line = state.estimator_classes[0]
        findings.append(Finding(
            "single_model_family", "low", cell, line, "Only one model family was tried",
            f"Every model in this notebook is from the '{next(iter(families))}' family and there is no dummy baseline. "
            "There is no universal best tabular algorithm, so a single family cannot show that this is a good choice.",
            "Screen a dummy baseline, a regularized linear model and one or two tree ensembles on identical CV folds, then keep the simplest model within noise of the best.",
            ["DCLAB-R13", "DCLAB-R14"], "no universal best algorithm model families", {"type": "finding"},
        ))
    if state.accuracy_sites and not (state.metrics_used & RICH_METRICS):
        cell, line = state.accuracy_sites[0]
        findings.append(Finding(
            "accuracy_only", "medium", cell, line, "Accuracy is the only metric",
            "Accuracy hides what matters when classes are imbalanced or errors have different costs: predicting the majority "
            "class can look excellent while missing every positive.",
            "Report ROC-AUC or average precision for ranking, log loss or Brier for probability quality, and precision/recall at the threshold you will actually use.",
            ["DCLAB-R18", "DCLAB-R14"], "metrics threshold decision cost ranking probability", {"type": "rule"},
        ))
    if state.timestamp_hint and state.random_splits and not state.time_aware_split:
        cell, line, _ = state.random_splits[0]
        findings.append(Finding(
            "random_split_with_time", "medium", cell, line, "Random split on data with timestamps",
            "The notebook handles dates or timestamps but splits rows at random. If the model will score future rows, a random "
            "split answers a different question; measure the forward-in-time score too.",
            "Sort by time and hold out the latest period (or use `TimeSeriesSplit`); report both numbers and explain the gap.",
            ["DCLAB-R02", "DCLAB-R19"], "random split time ordered holdout", {"type": "pitfall"},
        ))
    if state.classifier_used:
        for cell, line, call in state.random_splits:
            if _kw(call, "stratify") is None and not state.time_aware_split:
                findings.append(Finding(
                    "unstratified_split", "info", cell, line, "Split is not stratified",
                    "For classification, an unstratified split can leave the test set with a different class balance, especially for rare classes.",
                    "Pass `stratify=y` to `train_test_split` (unless you split by time or group on purpose).",
                    ["DCLAB-R02"], "split independence unit stratified", {"type": "rule"},
                ))
                break
    flagged_columns = set()
    blocked = blocked_columns()
    for key, (cell, line) in state.datasets_read.items():
        remaining = [c for c in blocked.get(key, []) if c not in state.dropped]
        named = [c for c in remaining if c in state.columns]
        unnamed = [c for c in remaining if c not in state.columns]
        if unnamed:
            flagged_columns.update(unnamed)
            precedent = next((known_leaks[c.lower()] for c in unnamed if c.lower() in known_leaks), None)
            findings.append(Finding(
                "blocked_columns_not_dropped", "high", cell, line, f"Known leakage columns are never dropped ({key})",
                f"This notebook loads the `{key}` data, whose decision-time contract forbids {', '.join(f'`{c}`' for c in unnamed)}. "
                "They are never dropped, so unless features are selected explicitly elsewhere they flow into the model "
                "and inflate every score.",
                "Drop them right after loading, e.g. `df = df.drop(columns=[" + ", ".join(repr(c) for c in unnamed) + "])`, "
                "and list allowed features explicitly.",
                ["DCLAB-R01", "DCLAB-R04"], " ".join(unnamed),
                {"record_id": precedent["record_id"]} if precedent else {"type": "leakage_precedent", "dataset": key},
            ))
        del named
    for column, (cell, line) in sorted(state.columns.items(), key=lambda kv: kv[1]):
        if column in state.dropped or column in flagged_columns:
            continue
        precedent = known_leaks.get(column.lower())
        if precedent:
            findings.append(Finding(
                "known_leakage_column", "high", cell, line, f"`{column}` is a known leakage column",
                f"`{column}` was measured as post-outcome leakage in the DCLab R&D campaigns ({precedent['title']}). "
                "It is not available at the moment a real prediction is made.",
                f"Drop `{column}` (and anything derived from it) from the model's inputs before splitting.",
                ["DCLAB-R04", "DCLAB-R01"], column, {"record_id": precedent["record_id"]},
            ))
        elif POST_OUTCOME_NAME.search(column) and len(column) <= 40:
            findings.append(Finding(
                "suspicious_column_name", "info", cell, line, f"Check when `{column}` is created",
                f"The name `{column}` suggests a value that may only exist after the outcome. A name is not proof; check when it is written.",
                f"Confirm that `{column}` exists at prediction time (data dictionary / source timestamp) before using it as a feature.",
                ["DCLAB-R05", "DCLAB-R01"], f"{column} post outcome leakage precedent", {"type": "leakage_precedent"},
            ))


# --------------------------------------------------------------------------- public API


def _known_leaks(index: EvidenceIndex) -> dict[str, dict[str, Any]]:
    leaks = {}
    for record in index.records:
        if record["type"] == "leakage_precedent":
            for column in record["metadata"].get("columns", []):
                leaks[str(column).lower()] = record
    return leaks


def _attach_proof(finding: Finding, index: EvidenceIndex) -> None:
    proof: list[dict[str, Any]] = []
    for rule_id in finding.rules:
        record = index.get(rule_id)
        if record:
            proof.append(record)
    filters = dict(finding.precedent_filters)
    if "record_id" in filters:
        record = index.get(filters["record_id"])
        if record:
            proof.append(record)
    elif finding.precedent_query:
        hits = index.search(finding.precedent_query, k=1, **filters)
        if hits and hits[0]["score"] > 0:
            proof.append(hits[0])
    unique: dict[str, dict[str, Any]] = {}
    for r in proof:
        unique.setdefault(r["record_id"], r)
    finding.proof = [
        {"record_id": r["record_id"], "type": r["type"], "title": r["title"], "text": r["text"], "citations": r["citations"]}
        for r in unique.values()
    ]


def review_cells(cells: list[dict[str, Any]], index: EvidenceIndex | None = None) -> dict[str, Any]:
    index = index or EvidenceIndex.load(ROOT)
    state = _State()
    findings: list[Finding] = []
    parse_errors = []
    for cell in cells:
        if cell["type"] != "code" or not cell["source"].strip():
            continue
        try:
            tree = ast.parse(_clean_source(cell["source"]))
        except SyntaxError as exc:
            parse_errors.append({"cell": cell["index"], "error": str(exc)})
            continue
        _analyze_cell(tree, cell["index"], state, findings)
    _notebook_level(state, findings, _known_leaks(index))

    deduped: dict[tuple[str, int, int], Finding] = {}
    for f in findings:
        deduped.setdefault((f.detector, f.cell, f.line), f)
    ordered = sorted(deduped.values(), key=lambda f: (f.cell, f.line, SEVERITY_ORDER[f.severity]))
    for finding in ordered:
        _attach_proof(finding, index)
    counts: dict[str, int] = {}
    for f in ordered:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return {
        "summary": {"findings": len(ordered), "by_severity": counts, "code_cells": sum(c["type"] == "code" for c in cells),
                    "split_cell": state.split_cell, "parse_errors": parse_errors},
        "findings": [f.to_json() for f in ordered],
        "limitations": [
            "Static analysis: code is never executed, so columns that only exist in the data are invisible.",
            "Detectors flag patterns; a human decides. A flagged pattern can be intentional and correct.",
        ],
    }


def review_notebook(path: Path | str, index: EvidenceIndex | None = None) -> dict[str, Any]:
    path = Path(path)
    cells = load_cells(path)
    report = review_cells(cells, index)
    report["notebook"] = path.name
    report["cells"] = cells
    return report


def review_source(source: str, index: EvidenceIndex | None = None) -> dict[str, Any]:
    """Review a single script or code string as one cell."""
    return review_cells([{"index": 0, "type": "code", "source": source}], index)


def iter_findings(report: dict[str, Any], cell: int) -> Iterable[dict[str, Any]]:
    return (f for f in report["findings"] if f["cell"] == cell)
