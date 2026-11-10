"""Independent development tests for scientific reliability and known failure modes (SCI-01).

Diagnoses and tests:
1. Segmentation feature selection: ensuring noise columns (e.g. region, signup_week) are consistently
   identified as unused and excluded across seeds without leaking into cluster formation.
2. Uplift evaluation: ensuring uplift models beat random targeting with positive confidence intervals
   and recover the responsive customer segment on holdout evaluations.
3. Regression against baseline: ensuring regression models on heavy-tailed, censored data beat the
   0.9 * median baseline threshold reliably.
"""

import unittest
import numpy as np

from dclab_core.replicas import customer_behaviours, campaign_uplift, delivery_time
from dclab_core import segment, uplift
from dclab_rnd import kinds
from dclab_rnd.kinds import acceptance


class ScientificFailuresDevTests(unittest.TestCase):
    def test_segmentation_feature_selection_excludes_noise_columns(self):
        """SCI-01: Columns no group uses must be declared unused and left out."""
        # Test across dev seeds that previously showed spurious noise splits
        for seed in (3, 4, 8, 9):
            replica = customer_behaviours.generate(seed=seed)
            cols = [c for c in replica.customers.columns if c != "customer_id"]
            # Test with default search seed 0 (as acceptance does)
            res = segment.segment(replica.customers, columns=cols, ks=range(2, 8), resamples=12, seed=0)
            unused = res["passes"]["unused_columns"]
            noise_cols = set(replica.facts["noise_columns"])
            self.assertTrue(noise_cols <= set(unused.keys()),
                            f"Seed {seed}: Expected noise columns {noise_cols} in unused {list(unused.keys())}")

    def test_uplift_evaluation_recovers_responsive_segment_and_beats_random(self):
        """SCI-01: Uplift model choice must recover responsive segment and beat random targeting."""
        for seed in (3, 4, 8):
            replica = campaign_uplift.generate(seed=seed)
            parts = uplift.split(replica.customers, treatment="got_offer", seed=0)
            features = [c for c in replica.customers.columns if c not in ("got_offer", "bought", "customer_id")]
            val = uplift.validate(replica.customers, parts, treatment="got_offer", outcome="bought", features=features, randomised=True)
            sc = uplift.score(replica.customers, parts, treatment="got_offer", outcome="bought", features=features, randomised=True, best=val["best"])
            
            # Must beat random targeting with positive confidence interval
            self.assertTrue(sc["best_beats_random"], f"Seed {seed}: best model {val['best']} did not beat random targeting")
            m = sc["models"][val["best"]]
            self.assertGreater(m["qini_coefficient_ci90"][0], 0.0, f"Seed {seed}: Qini CI90 lower bound not positive")
            
            # Responsive segment recovery in top tenth
            top_scores = np.array(sc["holdout_uplift"][val["best"]])
            top_idx = np.argsort(-top_scores)[:len(top_scores)//10]
            responsive = np.array(replica.facts["responsive"])[parts["holdout"]]
            recovery_rate = responsive[top_idx].mean()
            self.assertGreaterEqual(recovery_rate, 0.80, f"Seed {seed}: Responsive recovery rate {recovery_rate:.1%} < 80%")

    def test_regression_holdout_mae_beats_median_baseline(self):
        """SCI-01: Regression holdout MAE must be at most 0.9 times the training median baseline."""
        kind = next(k for k in kinds.all_kinds() if k.key == "regression")
        for seed in (3, 4, 8, 10, 732):
            res = acceptance.run(kind, seed=seed)
            missing = res.get("missing", [])
            self.assertEqual(missing, [], f"Seed {seed} regression acceptance failed: {missing}")


if __name__ == "__main__":
    unittest.main()
