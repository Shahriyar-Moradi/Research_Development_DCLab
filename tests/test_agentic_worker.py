"""Scientific tests for the existing ML environment (stdlib unittest runner)."""
import unittest
import numpy as np
import pandas as pd
from dclab_rnd.agentic.worker import DerivedFeatures, validate_plan, build_pipeline

def plan():
    return dict(dataset="bank_marketing", model="logistic_regression", parameters={}, drop_columns=[], stress_columns=[], features=[])

class WorkerSafety(unittest.TestCase):
    def test_blocked_ancestor_and_categorical_math(self):
        X=pd.DataFrame({"balance":[1.,2.],"duration":[10.,20.],"job":["a","b"]})
        p=plan();p["features"]=[dict(name="fe_duration",inputs=["duration"],operation="signed_log")]
        with self.assertRaises(ValueError): validate_plan(p,X,["job"])
        p["features"]=[dict(name="fe_job",inputs=["job"],operation="signed_log")]
        with self.assertRaises(ValueError): validate_plan(p,X,["job"])
    def test_zero_denominator_and_missing_parent(self):
        transform=DerivedFeatures([dict(name="fe_ratio",operation="ratio",inputs=["a","b"])])
        result=transform.fit_transform(pd.DataFrame({"a":[3.,2.,np.nan],"b":[0.,2.,3.]}))
        self.assertTrue(np.isnan(result.fe_ratio.iloc[0]));self.assertEqual(result.fe_ratio.iloc[1],1)
        self.assertTrue(np.isnan(result.fe_ratio.iloc[2]))
    def test_fold_only_imputer_and_unknown_category(self):
        p=plan();X=pd.DataFrame({"balance":[0.,10.,np.nan,20.],"job":["a","a","b","b"]})
        pipeline=build_pipeline(p,list(X),["job"]).fit(X,[0,1,0,1])
        before=pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
        pipeline.predict_proba(pd.DataFrame({"balance":[np.nan,999999.],"job":["never_seen","a"]}))
        after=pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
        np.testing.assert_array_equal(before,after);self.assertEqual(before[0],10)
    def test_unknown_or_all_dropped_features_rejected(self):
        p=plan();p["drop_columns"]=["unknown"]
        with self.assertRaises(ValueError):validate_plan(p,pd.DataFrame({"balance":[1]}),[])

if __name__=="__main__":unittest.main()
