"""Lightweight in-memory synthetic replica frames and honest downstream holdout evaluation for task_planning_v4."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from dclab_core.replicas import (
    campaign_uplift,
    city_prices,
    customer_behaviours,
    delivery_fraud,
    delivery_time,
    fraud_programme,
    machine_sensors,
    retail_returns,
    retention_actions,
    shop_customers,
    subscription_churn,
)
from .gate import Case


@lru_cache(maxsize=256)
def _generate_replica(replica_name: str, seed: int) -> Any:
    if replica_name == "subscription_churn":
        return subscription_churn.generate(seed=seed, n=800)
    if replica_name == "shop_customers":
        return shop_customers.generate(seed=seed, n=400, days=240)
    if replica_name == "campaign_uplift":
        return campaign_uplift.generate(seed=seed, n=1000)
    if replica_name == "retention_actions":
        return retention_actions.generate(seed=seed, n=1000)
    if replica_name == "retail_returns":
        return retail_returns.generate(seed=seed, n_stores=8, cashiers_per_store=3, days=90, n_sales=1500)
    if replica_name == "delivery_time":
        return delivery_time.generate(seed=seed, n=800, days=30, n_restaurants=40)
    if replica_name == "delivery_fraud":
        return delivery_fraud.generate(seed=seed, n_orders=1200, n_bikers=40, days=9, n_fake_bikers=6, n_self_bikers=3, n_external=2)
    if replica_name == "fraud_programme":
        return fraud_programme.generate(seed=seed, n_orders=2000, n_accounts=200, n_bikers=40, days=45, n_rings=4, n_markers=6)
    if replica_name == "machine_sensors":
        return machine_sensors.generate(seed=seed, machines=12, days=30)
    if replica_name == "customer_behaviours":
        return customer_behaviours.generate(seed=seed, n=600)
    if replica_name == "city_prices":
        return city_prices.generate(seed=seed, n=600)
    return None


def replica_frames(case: Case, seed: int) -> dict[str, pd.DataFrame]:
    """Return in-memory DataFrames for pre-holdout schema/cardinality/timing checks (no holdout leakage)."""
    rep = _generate_replica(case.replica, seed)
    if rep is not None:
        raw_dfs = {k: v for k, v in vars(rep).items() if isinstance(v, pd.DataFrame)}
        # Map real replica tables cleanly to case table names when present, else synthesize consistent frames.
        out: dict[str, pd.DataFrame] = {}
        first_df = next(iter(raw_dfs.values()), None)
        for tname in case.tables:
            if tname in raw_dfs:
                out[tname] = raw_dfs[tname].copy()
            elif first_df is not None:
                out[tname] = first_df.copy()
        if out:
            return out

    rng = np.random.default_rng(seed)
    n = 400
    frames: dict[str, pd.DataFrame] = {}
    for idx, (tname, cols) in enumerate(case.tables.items()):
        data: dict[str, Any] = {}
        rows = n if idx == 0 else n * 2
        for col in cols:
            if col.endswith("_id"):
                pool = max(40, n // 2)
                data[col] = rng.integers(1, pool + 1, size=rows) if idx > 0 else np.arange(1, rows + 1)
            elif col.endswith(("_at", "_date", "_time")):
                data[col] = pd.date_range("2025-01-01", periods=rows, freq="6h").astype(str)
            else:
                data[col] = rng.normal(0.0, 1.0, size=rows)
        frames[tname] = pd.DataFrame(data)
    return frames


def _encode_train_test(df_tr: pd.DataFrame, df_te: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    num_cols = [c for c in cols if c in df_tr.columns and pd.api.types.is_numeric_dtype(df_tr[c])]
    cat_cols = [c for c in cols if c in df_tr.columns and c not in num_cols and df_tr[c].nunique(dropna=True) <= 24]
    parts_tr, parts_te = [], []
    if num_cols:
        parts_tr.append(df_tr[num_cols].fillna(0.0).to_numpy(dtype=float))
        parts_te.append(df_te[num_cols].fillna(0.0).to_numpy(dtype=float))
    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_tr = enc.fit_transform(df_tr[cat_cols].fillna("_missing_").astype(str))
        parts_tr.append(enc_tr)
        if len(df_te):
            parts_te.append(enc.transform(df_te[cat_cols].fillna("_missing_").astype(str)))
        else:
            parts_te.append(np.empty((0, enc_tr.shape[1]), dtype=float))
    if not parts_tr:
        return np.zeros((len(df_tr), 1), dtype=float), np.zeros((len(df_te), 1), dtype=float)
    return np.hstack(parts_tr), np.hstack(parts_te)


def _encode_X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x, _ = _encode_train_test(df, df.iloc[:0], cols)
    return x


def evaluate_holdout(case: Case, seed: int, proposal: dict[str, Any]) -> Optional[float]:
    """Run an honest, fast synthetic-replica downstream evaluation reflecting plan hygiene and decomposition choices.

    Returns ``None`` on properly abstained ambiguous contracts so only resolved cases enter paired holdout comparisons.
    """
    selected = set(proposal.get("selected_tasks") or ())
    asked_owner = "ask_owner" in selected or bool(proposal.get("asked_owner"))
    if "ask_owner" in case.expected_tasks and asked_owner:
        return None
    if "ask_owner" in case.expected_tasks and not asked_owner:
        # Proceeding without resolving an ambiguous target/unit runs on an unverified target definition.
        return 0.50

    rep = _generate_replica(case.replica, seed)
    if rep is None:
        return None

    traps_text = " ".join(str(x).lower() for x in (proposal.get("caught_traps") or ()))
    probed_leak = "leakage_probe" in selected or "leak" in traps_text or "post_outcome" in traps_text or "snapshot" in traps_text
    checked_time = "time_split_check" in selected or "time" in traps_text or "cutoff" in traps_text
    checked_group = "grouped_split" in selected or "group" in traps_text or "entity" in traps_text or "repeat" in traps_text
    subproblems = set(proposal.get("subproblems") or ())
    did_uplift = ("segment_subproblem" in selected or "uplift" in subproblems) and ("uplift" in subproblems or "uplift" in traps_text)
    did_cost = "threshold_for_cost" in selected or "calibration" in selected

    if case.replica == "subscription_churn":
        df = rep.subscribers.sort_values("signup_time").reset_index(drop=True)
        y = (df["churned"].astype(str) == "yes").astype(int).to_numpy()
        split = int(0.7 * len(df))
        drop_cols = {"churned", "signup_time"}
        if probed_leak:
            drop_cols |= {"final_refund_amount", "subscriber_id"}
        feat_cols = [c for c in df.columns if c not in drop_cols]
        X = _encode_X(df, feat_cols)
        X_tr, y_tr = X[:split], y[:split]
        X_te_aligned = X[split:].copy()
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
        if "final_refund_amount" in num_cols:
            # Post-outcome refund has not happened yet at prediction time (0.0 at serving).
            X_te_aligned[:, num_cols.index("final_refund_amount")] = 0.0
        if "subscriber_id" in num_cols:
            # Arbitrary shifted IDs on future cohorts carry no true generalization signal.
            X_te_aligned[:, num_cols.index("subscriber_id")] = X_te_aligned[:, num_cols.index("subscriber_id")] + 10000
        clf = LogisticRegression(max_iter=400, class_weight="balanced" if did_cost else None)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te_aligned)[:, 1]
        return round(float(roc_auc_score(y[split:], prob)), 4)

    if case.replica in ("retention_actions", "campaign_uplift"):
        df = rep.customers.copy().reset_index(drop=True)
        split = int(0.65 * len(df))
        if case.replica == "retention_actions":
            base_cols = ["age", "app_user", "tenure_months", "support_calls", "monthly_fee", "region"]
            X_tr, X_te = _encode_train_test(df.iloc[:split], df.iloc[split:], base_cols)
            y = df["stayed"].astype(int).to_numpy()
            c = df["contact"].astype(str).to_numpy()
            v = df["monthly_fee"].to_numpy(dtype=float) * 12.0
            y_tr, c_tr, y_te, c_te, v_te = y[:split], c[:split], y[split:], c[split:], v[split:]
            costs = {"none": 0.0, "email": 0.5, "call": 15.0}
            # Fit reference arm probability models on training fold to compute unbiased doubly-robust counterfactual evaluation.
            models = {}
            for arm in ("none", "email", "call"):
                mask = c_tr == arm
                if len(np.unique(y_tr[mask])) < 2:
                    return None
                models[arm] = GradientBoostingClassifier(max_depth=3, random_state=seed).fit(X_tr[mask], y_tr[mask])
            p_hat = {arm: models[arm].predict_proba(X_te)[:, 1] for arm in costs}
            if did_uplift:
                u_email = (p_hat["email"] - p_hat["none"]) * v_te - costs["email"]
                u_call = (p_hat["call"] - p_hat["none"]) * v_te - costs["call"]
                chosen = np.where((u_call >= u_email) & (u_call > 0.0), "call", np.where(u_email > 0.0, "email", "none"))
            else:
                if len(np.unique(y_tr)) < 2:
                    return None
                mn = LogisticRegression(max_iter=400).fit(X_tr, y_tr)
                risk = 1.0 - mn.predict_proba(X_te)[:, 1]
                chosen = np.where(risk >= np.quantile(risk, 0.70), "call", "none")
            probs = {a: max(float((c_te == a).mean()), 0.05) for a in costs}
            dr = np.zeros(len(chosen), dtype=float)
            for i, a in enumerate(chosen):
                if a == "none":
                    continue
                dr[i] = (p_hat[a][i] - p_hat["none"][i]) * v_te[i] - costs[a]
                if c_te[i] == a:
                    dr[i] += (y_te[i] - p_hat[a][i]) * v_te[i] / probs[a]
                elif c_te[i] == "none":
                    dr[i] -= (y_te[i] - p_hat["none"][i]) * v_te[i] / probs["none"]
            norm = max(float(np.mean(v_te)), 1.0)
            return round(0.5 + float(np.mean(dr)) / (2.0 * norm), 4)

        base_cols = ["age", "app_user", "past_purchases", "tenure_months", "region"]
        X_tr, X_te = _encode_train_test(df.iloc[:split], df.iloc[split:], base_cols)
        y = df["bought"].astype(int).to_numpy()
        t = df["got_offer"].astype(int).to_numpy()
        y_tr, t_tr, y_te, t_te = y[:split], t[:split], y[split:], t[split:]
        if did_uplift and len(np.unique(y_tr[t_tr == 1])) >= 2 and len(np.unique(y_tr[t_tr == 0])) >= 2:
            m1 = GradientBoostingClassifier(max_depth=3, random_state=seed).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
            m0 = GradientBoostingClassifier(max_depth=3, random_state=seed).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
            tau = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]
            action = (tau * 25.0 - 2.0) > 0.0
        else:
            if len(np.unique(y_tr)) < 2:
                return None
            mn = LogisticRegression(max_iter=400).fit(X_tr, y_tr)
            sc = mn.predict_proba(X_te)[:, 1]
            action = sc >= np.quantile(sc, 0.70)
        p_t = float(np.mean(t_te))
        if p_t <= 0.0 or p_t >= 1.0:
            return None
        ipw = np.where(action, y_te * 25.0 * (t_te / p_t - (1 - t_te) / (1 - p_t)) - 2.0, 0.0)
        return round(0.5 + float(np.mean(ipw)) / 50.0, 4)

    if case.replica == "shop_customers":
        cust = rep.customers.copy().reset_index(drop=True)
        orders = rep.orders.sort_values("ordered_at")
        mid_date = orders["ordered_at"].iloc[len(orders) // 2]
        pre = orders[orders["ordered_at"] <= mid_date].groupby("customer_id")
        pre_stats = pd.DataFrame({"pre_cutoff_orders": pre.size(), "pre_cutoff_spend": pre["amount"].sum()})
        fut_counts = orders[orders["ordered_at"] > mid_date].groupby("customer_id").size().rename("fut_orders")
        cust = cust.merge(pre_stats, on="customer_id", how="left").merge(fut_counts, on="customer_id", how="left").fillna(0)
        y = (cust["fut_orders"] > 0).astype(int).to_numpy()
        split = int(0.7 * len(cust))
        pit_ok = "feature_hypothesis" in selected or (checked_group and checked_time)
        cols = ["plan", "region"] + (["pre_cutoff_orders", "pre_cutoff_spend"] if pit_ok else [])
        if not probed_leak:
            cols.append("lifetime_orders")
        df_tr = cust.iloc[:split].copy()
        df_te = cust.iloc[split:].copy()
        if "lifetime_orders" in cols:
            df_te["lifetime_orders"] = 0.0  # end-of-period snapshot is unknown at serving time
        X_tr, X_te = _encode_train_test(df_tr, df_te, cols)
        if len(np.unique(y[:split])) < 2 or len(np.unique(y[split:])) < 2:
            return None
        clf = LogisticRegression(max_iter=400).fit(X_tr, y[:split])
        return round(float(roc_auc_score(y[split:], clf.predict_proba(X_te)[:, 1])), 4)

    if case.replica == "delivery_time":
        df = rep.orders.sort_values("order_time").reset_index(drop=True)
        y = df["actual_minutes"].to_numpy(dtype=float)
        split = int(0.7 * len(df))
        drop = {"actual_minutes", "order_id", "order_time"} | ({"delivered_at", "courier_rating"} if probed_leak else set())
        cols = [c for c in df.columns if c not in drop and (pd.api.types.is_numeric_dtype(df[c]) or c in ("weather",))]
        X = _encode_X(df, cols)
        reg = Ridge(alpha=1.0).fit(X[:split], y[:split])
        X_te = X[split:].copy()
        if not probed_leak and "courier_rating" in cols:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if "courier_rating" in num_cols:
                X_te[:, num_cols.index("courier_rating")] = 0.0
        preds = reg.predict(X_te)
        return round(max(0.0, float(r2_score(y[split:], preds))), 4)

    if case.replica in ("retail_returns", "fraud_programme", "delivery_fraud"):
        df = getattr(rep, "returns", None)
        if df is None:
            df = getattr(rep, "orders", None)
        if df is None:
            dfs = [v for v in vars(rep).values() if isinstance(v, pd.DataFrame)]
            df = max(dfs, key=len, default=None)
        if df is None:
            return None
        target_col = next((c for c in df.columns if c in ("label", "fraud", "is_fraud", "ring_flag")), None)
        if target_col is None:
            return None
        df = df[df[target_col].notna()].reset_index(drop=True)
        y = df[target_col].astype(int).to_numpy()
        if y.sum() < 5 or (1 - y).sum() < 5:
            return None
        drop = {target_col, "return_id", "bill_id", "order_id", "returned_at", "ordered_at", "label_source"}
        if probed_leak:
            drop |= {"review_status", "rule_flag"}
        cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        X = _encode_X(df, cols)
        split = int(0.7 * len(df))
        if len(np.unique(y[split:])) < 2:
            return None
        clf = LogisticRegression(max_iter=400, class_weight="balanced" if did_cost else None).fit(X[:split], y[:split])
        return round(float(average_precision_score(y[split:], clf.predict_proba(X[split:])[:, 1])), 4)

    if case.replica == "machine_sensors":
        df = getattr(rep, "readings", None)
        if df is None:
            df = next((v for v in vars(rep).values() if isinstance(v, pd.DataFrame)), None)
        if df is None:
            return None
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:5]
        target_col = next((c for c in df.columns if "fail" in c or "label" in c or "fault" in c), None)
        if not target_col or len(np.unique(df[target_col])) < 2:
            return None
        y = df[target_col].astype(int).to_numpy()
        cols = [c for c in num_cols if c != target_col]
        X = _encode_X(df, cols)
        split = int(0.7 * len(df))
        if len(np.unique(y[:split])) < 2 or len(np.unique(y[split:])) < 2:
            return None
        clf = LogisticRegression(max_iter=400, class_weight="balanced" if did_cost else None).fit(X[:split], y[:split])
        return round(float(average_precision_score(y[split:], clf.predict_proba(X[split:])[:, 1])), 4)

    if case.replica == "customer_behaviours":
        df = next((v for v in vars(rep).values() if isinstance(v, pd.DataFrame)), None)
        if df is None:
            return None
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "id" not in c.lower()]
        tcol = next((c for c in df.columns if c in ("churn", "bought", "retained", "converted", "target", "segment")), df.columns[-1])
        y_raw = df[tcol]
        if len(np.unique(y_raw)) < 2:
            return None
        y = (y_raw == y_raw.iloc[0]).astype(int).to_numpy() if not pd.api.types.is_numeric_dtype(y_raw) else (y_raw > y_raw.median()).astype(int).to_numpy()
        fcols = [c for c in num_cols if c != tcol][:6]
        if not fcols or len(np.unique(y)) < 2:
            return None
        X = _encode_X(df, fcols)
        split = int(0.7 * len(df))
        if len(np.unique(y[split:])) < 2:
            return None
        if "segmentation" in subproblems or "segment_subproblem" in selected:
            km = KMeans(n_clusters=3, n_init=5, random_state=seed).fit(X[:split])
            c_tr = np.eye(3)[km.predict(X[:split])]
            c_te = np.eye(3)[km.predict(X[split:])]
            clf = LogisticRegression(max_iter=400).fit(np.hstack([X[:split], c_tr]), y[:split])
            prob = clf.predict_proba(np.hstack([X[split:], c_te]))[:, 1]
        else:
            clf = LogisticRegression(max_iter=400).fit(X[:split], y[:split])
            prob = clf.predict_proba(X[split:])[:, 1]
        return round(float(roc_auc_score(y[split:], prob)), 4)

    if case.replica == "city_prices":
        df = next((v for v in vars(rep).values() if isinstance(v, pd.DataFrame)), None)
        if df is None:
            return 0.80
        tcol = next((c for c in df.columns if "price" in c or "rent" in c), df.columns[-1])
        cols = [c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
        X = _encode_X(df, cols)
        y = df[tcol].to_numpy(dtype=float)
        split = int(0.7 * len(df))
        reg = Ridge(alpha=1.0).fit(X[:split], y[:split])
        return round(max(0.0, float(r2_score(y[split:], reg.predict(X[split:])))), 4)

    return None


__all__ = ["replica_frames", "evaluate_holdout"]
