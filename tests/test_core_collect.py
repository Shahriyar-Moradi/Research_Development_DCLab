"""34.14, what to collect next. A churn replica whose outcome rests on tenure and support contacts, offered without the
support contacts: support is the first usual kind named missing, and the advice says it is prior knowledge, not measured
(with the outcome shuffled the missing kinds are the same: they never come from the data). A planted usage gap names
usage instead. What each present group adds is read with an interval: tenure adds; a group the outcome never used is not
named. Names are matched by whole word parts (the review of 34.14: "count" in account_age, "ip" in shipping, "week" in
sales_last_week, TechSupport as a support contact). A segment from a real learning curve names its column, never its level,
in the words that leave the record.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    from dclab_core import collect  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)

PROBLEM = "Predict which subscribers will churn next month"


def churn(n=3000, seed=0, with_usage=True):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"tenure_months": rng.integers(1, 72, n), "plan": rng.integers(0, 3, n), "monthly_fee": rng.uniform(5, 60, n),
                          "late_payments": rng.poisson(0.4, n), "support_calls": rng.poisson(1.2, n), "logins_last_month": rng.poisson(8, n)})
    logit = -0.04 * table["tenure_months"] + 0.9 * table["support_calls"] + 0.3 * table["late_payments"] - 0.08 * table["logins_last_month"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).to_numpy().astype(int)
    return (table if with_usage else table.drop(columns=["logins_last_month"])), y


def predictor(X, y):
    """Out-of-fold predictions on the training folds, with some columns made constant."""
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))

    def predict_without(columns):
        Z = X.copy()
        for c in columns:
            Z[c] = 0.0
        out = np.zeros(len(Z))
        for fit, valid in folds:
            out[valid] = LogisticRegression(max_iter=500).fit(Z.iloc[fit], y[fit]).predict_proba(Z.iloc[valid])[:, 1]
        return out
    return predict_without


def advise(X, y, **kw):
    return collect.advise(list(X.columns), problem=PROBLEM, y=y, predict_without=predictor(X, y), metric_fn=roc_auc_score,
                          metric="roc_auc", bootstrap=150, **kw)


@unittest.skipIf(MISSING, MISSING)
class CollectTests(unittest.TestCase):
    def test_the_missing_kind_is_named_first_and_said_to_be_prior_knowledge(self):
        table, y = churn()
        offered = table.drop(columns=["support_calls"])  # the support system's export was never joined
        advice = advise(offered, y)
        self.assertEqual(advice.domain, "churn")
        self.assertEqual(advice.missing[0].kind, "support")
        self.assertIn("support contacts", advice.plain)
        self.assertIn("not measured", advice.plain)
        shuffled = advise(offered, np.random.default_rng(9).permutation(y))
        self.assertEqual([m.kind for m in shuffled.missing], [m.kind for m in advice.missing])  # never from the data
        no_usage, y2 = churn(with_usage=False)
        planted = collect.advise(list(no_usage.columns), problem=PROBLEM)
        self.assertEqual(planted.missing[0].kind, "usage")  # a planted usage gap names usage
        self.assertNotIn("support", [m.kind for m in planted.missing])

    def test_what_a_group_adds_is_read_with_an_interval_and_noise_is_not_named(self):
        table, y = churn()
        advice = advise(table, y)
        by = {g.group: g for g in advice.groups}
        self.assertEqual(by["tenure"].verdict, "adds")
        self.assertGreater(by["tenure"].low, 0)
        self.assertNotEqual(by["contract"].verdict, "adds")  # the plan never moved the outcome
        self.assertNotEqual(by["billing"].verdict, "adds")  # nor the fee
        self.assertIn("the roc_auc falls by", advice.plain)  # the sentence reads the right way round
        self.assertIn("beyond the other groups", advice.plain)
        self.assertIn("reads as little for each", advice.assumption)
        self.assertEqual(advice.schema_version, "dclab_core.collect_advice.v1")

    def test_names_are_matched_by_whole_word_parts(self):
        cases = {
            ("account_age_days", "fraud"): "account_age", ("country", "fraud"): "location", ("shipping_city", "fraud"): "location",
            ("zip_code", "fraud"): "location", ("ip_address", "fraud"): "device", ("accounts_per_device", "fraud"): "identity_links",
            ("sales_last_week", "demand"): "history", ("rolling_7day_sales", "demand"): "history", ("week_of_year", "demand"): "calendar",
            ("TechSupport", "churn"): "contract", ("PaymentMethod", "churn"): "billing", ("days_since_last_login", "churn"): "usage",
            ("support_calls", "churn"): "support", ("late_payments", "churn"): "payments", ("tenure", "churn"): "tenure",
            ("MARRIAGE", "credit"): None, ("PAY_0", "credit"): "history", ("BILL_AMT1", "credit"): "debt", ("AGE", "credit"): "age",
        }
        for (name, domain), kind in cases.items():
            self.assertEqual(collect.kind_of(name, domain), kind, (name, domain))
        self.assertEqual(collect.domain_of("Predict which invoices are paid late"), None)  # not a delivery
        self.assertEqual(collect.domain_of("Which hotel bookings will be cancelled"), None)  # not a churn
        self.assertEqual(collect.domain_of("Find fake deliveries among the orders"), "fraud")
        self.assertEqual(collect.domain_of(PROBLEM), "churn")

    def test_four_more_domains_are_named_and_matched(self):
        """0.117.0 (the plan's item 13): maintenance, conversion, claims and returns, after the first five."""
        problems = {"Alert before a machine breaks down, from its sensor readings": "maintenance",
                    "Which customers respond to the discount campaign": "conversion",
                    "Flag an insurance claim worth a closer look": "claims",
                    "Which orders will be returned for a refund": "returns",
                    "Predict which subscribers churn after a failed payment": "churn",  # the first five keep their problems
                    "Predict the return on a stock next month": None, "Estimate the time to lead a project": None,
                    "Which machine learning model fits": None, "Predict the output of the machine": None}
        for text, domain in problems.items():
            self.assertEqual(collect.domain_of(text), domain, text)
        cases = {("vibration_rms", "maintenance"): "sensors", ("hours_since_last_service", "maintenance"): "maintenance_history",
                 ("alarm_count", "maintenance"): "errors", ("email_opens_30d", "conversion"): "engagement",
                 ("utm_source", "conversion"): "channel", ("prior_claims", "claims"): "claim_history",
                 ("annual_premium", "claims"): "policy", ("past_returns", "returns"): "customer_history",
                 ("product_category", "returns"): "item"}
        for (name, domain), kind in cases.items():
            self.assertEqual(collect.kind_of(name, domain), kind, (name, domain))
        out = collect.advise(["machine_id", "operating_hours", "alarm_count"], problem="Predict machine breakdowns")
        missing = [m.kind for m in out.missing] if hasattr(out, "missing") else [m["kind"] for m in out["missing"]]
        self.assertEqual(missing[0], "sensors")  # the most informative kind no column holds, named first

    def test_a_real_segment_names_its_column_never_its_level_and_no_domain_weighs_columns_one_by_one(self):
        segment = {"column": "plan", "level": "basic_monthly", "rows": 400, "row_share": 0.2, "error_share": 0.4, "answer": "likely_help",
                   "reason": "rises"}
        table, y = churn()
        advice = collect.advise(["x1", "x2"], problem="Predict the output of the machine", data_value={"segments": [segment]},
                                data_needs=[{"question": "'Churn' needs the support tickets table, but no uploaded table is like it."}])
        self.assertIsNone(advice.domain)
        self.assertEqual(advice.missing, [])
        self.assertEqual((advice.label_segments[0].column, advice.label_segments[0].level), ("plan", "basic_monthly"))
        self.assertIn("segments of plan", advice.plain)
        self.assertNotIn("basic_monthly", advice.plain)  # a level is a value of the table: it stays in the record
        self.assertIn("may help", advice.plain)
        self.assertEqual(len(advice.questions), 1)
        loose = collect.advise(list(table.columns), problem="Predict the output of the machine", y=y, predict_without=predictor(table, y),
                               metric_fn=roc_auc_score, metric="roc_auc", bootstrap=60)
        self.assertEqual({g.group for g in loose.groups}, set(table.columns))  # with no domain, each column on its own


if __name__ == "__main__":
    unittest.main()
