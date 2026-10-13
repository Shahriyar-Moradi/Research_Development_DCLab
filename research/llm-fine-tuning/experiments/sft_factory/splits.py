"""The one place that says which dataset or scenario belongs to which split.

Every generator in sft_factory imports this, so a dataset can never land in two splits.

* validation: three sources, so checkpoint choice does not rest on one dataset: `wine_quality` and
  `heart_disease` (public) and the `delivery_time` replica (front-of-workflow stages);
* test: `online_shoppers`, `german_credit` (public) and the `product_reviews`, `city_prices` replicas.
"""

VAL = ("wine_quality", "heart_disease", "delivery_time")
TEST = ("online_shoppers", "german_credit", "product_reviews", "city_prices")

SPLIT_OF = {**{k: "val" for k in VAL}, **{k: "test" for k in TEST}}


def split_of(key: str) -> str:
    return SPLIT_OF.get(key, "train")
