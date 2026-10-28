"""Variant generators for challenge packs (H0 of the harness-learning plan; Phase 9's 31.3).

``generate(family, seed)`` draws one invented table with its traps planted and returns it with its pack (what a
solver reads) and its key (what the grader reads). The same family and seed give the same table file, byte for byte;
the pack's id is opaque (a fresh token unless one is given), and nothing public carries the seed.

The first family, ``groups`` (duplicate and group leakage): each entity (a customer, a patient, a borrower, an account)
has several rows, and a lasting risk of its own that the row's features explain only in part, so a random split lets
validation score rows of the entities it trained on. The knobs, drawn from the seed:

* the domain skin (retail returns, clinic visits, loans, support tickets) and the task (binary or regression); a sealed
  variant draws from :data:`SEALED_SKINS` instead, eight domains written by an independent agent, so no column name a
  learner practised on reaches the exam (LRN-008 found names memorised across skins);
* how many entities and rows per entity, and a share of exact copies of an entity's rows;
* whether the entity's key has a key-like name (``customer_id``) or a plain one (``household``): a rule that finds
  group keys by name sees only the first;
* a column written after the outcome, named with a word that gives it away (``refund_settled``) or plainly
  (``adjustment_amount``);
* a decoy: a column known before the moment but named like a leak (``prior_refund_count``);
* a time column or none.

Every table is invented from the seed; no row comes from a person or a file.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .challenge_pack import Brief, ChallengeKey, ChallengePack, partition_of

VERSION = "variants_v1"


@dataclass(frozen=True)
class Variant:
    table: pd.DataFrame
    csv: bytes                # the table file's exact bytes: what the pack's SHA-256 is of, and what a pack folder holds
    pack: ChallengePack
    key: ChallengeKey


@dataclass(frozen=True)
class Skin:
    entity_id: str        # a key-like name
    entity_plain: str     # a plain name for the same key
    prefix: str           # the key's values
    row: str              # what one row is
    row_key: str
    binary: str           # the outcome, binary
    regression: str       # the outcome, a number
    leak_worded: str      # written after the outcome, a giving-away word in its name
    leak_plain: str       # written after the outcome, named plainly
    decoy: str            # known before the moment, named like a leak
    attribute: str        # an entity-level category
    amount: str           # a row-level number
    channel: str          # a row-level category
    created: str          # the time column


SKINS = (
    Skin("customer_id", "household", "HH", "order", "order_id", "returned", "return_value", "refund_settled", "adjustment_amount",
         "prior_refund_count", "loyalty_tier", "basket_value", "sales_channel", "ordered_at"),
    Skin("patient_id", "patient", "PT", "visit", "visit_id", "readmitted", "length_of_stay", "discharge_final_code", "ward_transfer_count",
         "prior_resolved_visits", "insurance_plan", "lab_score", "admission_route", "admitted_at"),
    Skin("borrower_id", "borrower", "BR", "loan", "loan_id", "defaulted", "loss_amount", "closed_balance", "recovery_amount",
         "prior_closed_loans", "employment_band", "requested_amount", "application_channel", "applied_at"),
    Skin("account_no", "account", "AC", "ticket", "ticket_id", "escalated", "resolution_hours", "resolved_satisfaction", "handover_count",
         "prior_resolved_tickets", "support_plan", "message_length", "contact_channel", "opened_at"),
)


SEALED_SKINS = (  # written by an independent agent that read no file of this repository (2026-10-29): sealed variants only
    Skin("policyholder_id", "policyholder", "PH", "claim", "claim_number", "claim_denied", "days_to_payout", "settled_payout_amount", "adjuster_hours", "prior_denied_claims", "coverage_level", "claimed_amount", "filing_channel", "filed_at"),
    Skin("flyer_number", "traveler", "FF", "reservation", "pnr_code", "no_show", "rebooking_fee", "refunded_fare_total", "voucher_value", "past_cancelled_trips", "loyalty_status", "ticket_fare", "booking_platform", "reserved_at"),
    Skin("subscriber_no", "subscriber", "SB", "contract", "contract_id", "ported_out", "data_overage_gb", "final_bill_amount", "service_credit_eur", "previous_disconnections", "plan_tier", "monthly_charge", "activation_channel", "activated_at"),
    Skin("shipper_code", "shipper", "SH", "shipment", "waybill_no", "delivered_late", "hours_delayed", "post_delivery_claim_value", "handling_surcharge", "past_lost_parcels", "account_grade", "declared_weight_kg", "pickup_mode", "dispatched_at"),
    Skin("learner_id", "learner", "LN", "enrollment", "enrollment_no", "dropped_out", "modules_completed", "final_exam_grade", "tutor_minutes", "prior_withdrawn_courses", "subscription_level", "course_fee", "signup_source", "enrolled_at"),
    Skin("patron_id", "patron", "PT", "session", "session_id", "session_missed", "minutes_attended", "penalty_after_absence", "trainer_rating", "past_missed_sessions", "membership_plan", "session_price", "booking_method", "requested_at"),
    Skin("renter_code", "renter", "RN", "rental", "agreement_no", "returned_damaged", "extra_km_driven", "damage_charge_settled", "fuel_topup_litres", "prior_damage_reports", "driver_tier", "daily_rate", "reservation_source", "rental_started_at"),
    Skin("debtor_code", "client_company", "AC", "invoice", "invoice_number", "paid_late", "days_past_due", "collected_post_due", "discount_taken", "past_written_off_invoices", "credit_band", "invoice_total", "billing_channel", "issued_at"),
)
SEALED_SKINS_SHA256 = "4b5bad0285e577f0d8c7867c66af1dfdf7a06516909af5e7208580673b20a31c"  # the sealed skins pinned: a change is a new sealed set

def token() -> str:
    """An opaque pack id: never derived from the seed, so a solver cannot draw the key again from it."""
    return f"pk-{secrets.token_hex(6)}"


def groups(seed: int, pack: str | None = None) -> Variant:
    """One variant of the duplicate and group leakage family; ``pack`` its id (a fresh token by default)."""
    rng = np.random.default_rng(seed)
    sealed = partition_of("groups", seed) == "sealed"
    pool = SEALED_SKINS if sealed else SKINS  # the exam's domains are its own: no name a learner practised on reaches it
    skin = pool[int(rng.integers(len(pool)))]
    task = "binary" if rng.random() < 0.7 else "regression"
    key_name = "id" if rng.random() < 0.5 else "plain"
    leak_name = "worded" if rng.random() < 0.5 else "plain"
    duplicates = float(rng.choice([0.0, 0.02, 0.05]))
    with_decoy = bool(rng.random() < 0.6)
    timed = bool(rng.random() < 0.4)
    entities = int(rng.integers(150, 300))
    per_entity = rng.integers(2, 11, entities)

    risk = rng.normal(0.0, 1.0, entities)                      # each entity's lasting risk, unseen in any column
    tier = rng.choice(["bronze", "silver", "gold", "platinum"], entities)
    age = rng.integers(30, 3000, entities)                     # nearly unique per entity: a model can learn entities through it
    entity_of = np.repeat(np.arange(entities), per_entity)
    n = len(entity_of)
    amount = np.round(rng.lognormal(3.5, 0.6, n), 2)
    items = rng.poisson(3.0, n) + 1
    channel = rng.choice(["web", "phone", "store"], n, p=[0.5, 0.3, 0.2])
    z = (np.log(amount) - 3.5) / 0.6
    tier_effect = pd.Series(tier).map({"bronze": 0.6, "silver": 0.1, "gold": -0.3, "platinum": -0.7}).to_numpy()[entity_of]
    # what the row's own features explain (an honest model learns it) beside the entity's risk (a random split leaks it)
    signal = 1.0 * risk[entity_of] + 0.9 * z + 0.5 * (items - 4) / 2 + tier_effect + np.where(channel == "phone", 0.6, 0.0)
    if task == "binary":
        y = (rng.random(n) < 1 / (1 + np.exp(-(signal - 1.0)))).astype(int)
        leak = np.where(y == 1, np.round(amount * rng.uniform(0.3, 1.0, n), 2), 0.0)
        flip = rng.random(n) < 0.03                            # a few written wrong, as real columns are
        leak = np.where(flip, np.round(amount * rng.uniform(0.3, 1.0, n), 2) * (1 - y), leak)
        target = skin.binary
    else:
        y = np.round(40 + 12 * signal + rng.normal(0, 6, n), 2)
        leak = np.round(y * (1 + rng.normal(0, 0.03, n)), 2)
        target = skin.regression

    key_column = skin.entity_id if key_name == "id" else skin.entity_plain
    leak_column = skin.leak_worded if leak_name == "worded" else skin.leak_plain
    frame = pd.DataFrame({
        skin.row_key: [f"R{v:09d}" for v in rng.choice(10**9, n, replace=False)],  # random values: never the seed
        key_column: [f"{skin.prefix}-{e:05d}" for e in entity_of],
        skin.attribute: tier[entity_of],
        "account_age_days": age[entity_of],
        skin.amount: amount,
        "item_count": items,
        skin.channel: channel,
        leak_column: leak,
        target: y,
    })
    if timed:  # each entity is active for some weeks of the year, as customers and patients come and go: a forward split
        start = rng.integers(0, 330, entities)                  # can still hold out entities it never saw (a time-and-group split)
        days = start[entity_of] + rng.integers(0, 36, n)
        frame.insert(1, skin.created, (pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d"))
    if timed:  # the decoy counts an entity's earlier rows: earlier in time where there is a time
        frame = frame.sort_values(skin.created, kind="stable").reset_index(drop=True)
    if with_decoy:  # known before the moment: the entity's earlier rows only
        frame[skin.decoy] = frame.groupby(key_column).cumcount().astype(int)
    copies = int(round(duplicates * len(frame)))
    if copies:
        frame = pd.concat([frame, frame.iloc[rng.choice(len(frame), copies, replace=False)]], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=int(rng.integers(2**31 - 1))).reset_index(drop=True) if not timed else \
        frame.sort_values(skin.created, kind="stable").reset_index(drop=True)

    pid = pack or token()
    if task == "binary":
        request = f"Which {skin.row}s will be {target}? Score each {skin.row} so we can act on the riskiest ones."
    else:
        request = f"Predict the {target.replace('_', ' ')} of each {skin.row} so we can plan for it."
    brief = Brief(request=request, target=target, task=task,
                  moment=f"when the {skin.row} is created, before anything about its outcome is known")
    csv = frame.to_csv(index=False).encode("utf-8")
    pack_ = ChallengePack(id=pid, family="groups", brief=brief, columns=list(frame.columns), rows=len(frame),
                          table_sha256=hashlib.sha256(csv).hexdigest())
    honest = [c for c in frame.columns if c not in {target, leak_column, key_column, skin.row_key, skin.decoy, skin.created}]
    key = ChallengeKey(pack=pid, family="groups", seed=seed, partition=partition_of("groups", seed), leaks=[leak_column],
                       group_column=key_column, row_key=skin.row_key, decoys=[skin.decoy] if with_decoy else [],
                       known_before=honest, time_column=skin.created if timed else None,
                       duplicate_rows=copies,
                       knobs={"skin": skin.row, **({"skins": "sealed_v1"} if sealed else {}), "task": task, "key_name": key_name, "leak_name": leak_name, "duplicates": duplicates,
                              "decoy": with_decoy, "timed": timed, "entities": entities, "rows": len(frame), "generator": VERSION})
    return Variant(frame, csv, pack_, key)


FAMILIES: dict[str, Callable[..., Variant]] = {"groups": groups}


def generate(family: str, seed: int, pack: str | None = None) -> Variant:
    if family not in FAMILIES:
        raise KeyError(f"no variant family {family!r} (families: {', '.join(FAMILIES)})")
    return FAMILIES[family](int(seed), pack)


SHARED = ("account_age_days", "item_count")  # columns every skin of the groups family has


def family_names(family: str) -> list[str]:
    """Every name the family's generator can give a column or a key's values, across all its skins: a lesson for the
    whole family must name none of them (a skin's names repeat across the learning, check and sealed variants)."""
    if family != "groups":
        raise KeyError(f"no names known for the family {family!r}")
    return sorted({getattr(s, f) for s in (*SKINS, *SEALED_SKINS) for f in Skin.__dataclass_fields__} | set(SHARED))


def seeds(family: str, partition: str, count: int, start: int = 0) -> list[int]:
    """The first ``count`` seeds of a family that fall in a partition."""
    out, seed = [], start
    while len(out) < count:
        if partition_of(family, seed) == partition:
            out.append(seed)
        seed += 1
    return out


def reference(key: ChallengeKey) -> dict[str, Any]:
    """What a person who knows the key would state on the Solution page: the leak forbidden, the row key an identifier,
    the entity as the group (and the time column when there is one). Graded, it must pass: a variant it fails is not
    solvable, and is dropped from every set."""
    statements: dict[str, Any] = {"forbidden": [{"column": c, "reason": "written after the outcome"} for c in key.leaks],
                                  "identifiers": [key.row_key] if key.row_key else [], "group_column": key.group_column}
    if key.time_column:
        statements["time_column"] = key.time_column
    return statements


__all__ = ["VERSION", "Variant", "Skin", "SKINS", "SEALED_SKINS", "SEALED_SKINS_SHA256", "SHARED", "family_names", "token", "groups", "FAMILIES", "generate", "seeds", "reference"]
