"""End-to-end comparison (2026-11-18): DCLab's rules, DCLab with a model as its agent, and the same model alone with a
Python tool, each given one table and one request, scored by code against a future set no arm sees. Plan:
Dev_Plans/End_to_End_Comparison_2026-11-18/PLAN.md."""

CAMPAIGN = "e2e_v1"
MODEL = "qwen/qwen3.8-flash"  # the product's model; never a Claude model (the owner, 2026-11-15)
DEV_SEEDS = (11, 12, 13)
HELD_OUT_SEEDS = (101, 102, 103, 104)  # read once, by the gate, after the owner reads the dev result: never in this plan
KINDS = ("ticket_routing", "delivery_time")
FUTURE_SEED_OFFSET = 1000  # the future set is the same replica drawn with another seed: the same world, new cases
