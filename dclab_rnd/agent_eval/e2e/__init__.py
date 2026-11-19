"""End-to-end comparison (2026-11-18): DCLab's rules, DCLab with a model as its agent, and the same model alone with a
Python tool, each given one table and one request, scored by code against a future set no arm sees. Plan:
Dev_Plans/End_to_End_Comparison_2026-11-18/PLAN.md."""

CAMPAIGN = "e2e_v1"
MODEL = "qwen/qwen3.8-flash"  # the product's model; never a Claude model (the owner, 2026-11-15)
DEV_SEEDS = (11, 12, 13)
HELD_OUT_SEEDS = (101, 102, 103, 104)  # read once, by the gate, after the owner reads the dev result: never in this plan
KINDS = ("ticket_routing", "delivery_time")
FUTURE_SEED_OFFSET = 1000  # e2e_v1's future set ("new_seed"): the replica drawn again with another seed
# 2026-11-19: the default future ("continued") is the latest rows of one replica, kept back by time. With another seed,
# every restaurant kept its number but got new behaviour (a restaurant's mean time correlated -0.02 between the table
# and the future, 0.42 between early and late rows of one table), so an arm that learned restaurant 17 was punished for
# knowledge that holds in real use. E2E-001 to E2E-007, the held-out gate among them, were scored on "new_seed".
FUTURE = "continued"
