Your ExpertTrace JSON had workflow ordering problems checked by `dclab_core.expert.workflow_problems`:
${problems}
Please reply with the corrected full JSON object ensuring: (1) a `leakage` stage step (WF-05) comes before `features` (WF-06) and `models` (WF-07), and (2) `final` holdout evaluation (WF-09) appears once at the end after `models`.
