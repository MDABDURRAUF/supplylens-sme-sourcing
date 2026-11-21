"""Stress tests: what a tariff change, a lost supplier or a demand change does to the plan.

Each test compares the current plan with what happens if nothing is changed
("stay") and with a re-optimized plan ("adapt"). All differences are planning
estimates under the stated assumptions, not realized savings.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .optimizer import Result, Settings, evaluate_plan, optimize, prepare


def with_duty_change(demand: pd.DataFrame, offers: pd.DataFrame, country: str, added_rate: float,
                     settings: Settings = Settings(), tariffs=None) -> pd.DataFrame:
    """Offers with the duty rate on ``country`` raised by ``added_rate`` (0.10 = ten points).

    Duty rates are first resolved from ``tariffs``, so the returned table has
    an explicit ``duty_rate`` for every row and needs no tariff table. A
    negative ``added_rate`` lowers the duty, never below zero.
    """
    resolved = prepare(demand, offers, settings, tariffs, apply_deadline=False)[1][["sku", "supplier", "country", "duty_rate"]]
    out = offers.drop(columns=[c for c in ("country", "duty_rate") if c in offers.columns]).copy()
    for col in ("sku", "supplier"):
        out[col] = out[col].astype(str).str.strip()
    out = out.merge(resolved, on=["sku", "supplier"], how="inner")
    hit = out["country"] == str(country).strip()
    out.loc[hit, "duty_rate"] = (out.loc[hit, "duty_rate"] + added_rate).clip(lower=0.0)
    return out


def baseline_lowest_price(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings = Settings(), tariffs=None) -> Result:
    """The plan a buyer gets by choosing on quoted price alone, costed in full.

    Same rules as the optimized plan; only the objective differs. Comparing it
    with ``optimize`` shows what duty, freight, fixed costs, financing and
    delivery risk add to the cheapest-looking option.
    """
    return optimize(demand, offers, settings, tariffs, objective="unit_price")


def tariff_shock(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, country: str, added_rate: float,
                 tariffs=None) -> dict:
    """Effect of raising the duty on ``country`` by ``added_rate``.

    Returns ``current`` (today's optimal plan), ``stay`` (the same plan costed
    at the new duty), ``adapt`` (re-optimized at the new duty) and:

    ``exposure``           extra cost if the plan is left unchanged
    ``adaptation_saving``  how much of that is avoided by re-optimizing
    ``net_impact``         extra cost after re-optimizing
    """
    current = optimize(demand, offers, settings, tariffs)
    if current.status != "optimal":
        return {"current": current, "stay": None, "adapt": None, "exposure": float("nan"),
                "adaptation_saving": float("nan"), "net_impact": float("nan")}
    shocked = with_duty_change(demand, offers, country, added_rate, settings, tariffs)
    stay = evaluate_plan(demand, shocked, current.allocations, settings)
    adapt = optimize(demand, shocked, settings)
    exposure = stay.total_expected_cost - current.total_expected_cost
    net = adapt.total_expected_cost - current.total_expected_cost if adapt.status == "optimal" else float("nan")
    return {"current": current, "stay": stay, "adapt": adapt, "exposure": exposure,
            "adaptation_saving": exposure - net, "net_impact": net}


def supplier_outage(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, supplier: str, tariffs=None) -> dict:
    """Effect of losing ``supplier`` entirely.

    Returns ``current``, ``adapt`` (best plan without the supplier, which may
    be infeasible), ``units_affected`` (units the current plan buys from the
    supplier) and ``extra_cost`` (NaN when no plan is possible without it).
    """
    current = optimize(demand, offers, settings, tariffs)
    remaining = offers[offers["supplier"].astype(str).str.strip() != str(supplier).strip()]
    adapt = optimize(demand, remaining, settings, tariffs)
    a = current.allocations
    units = float(a.loc[a["supplier"] == str(supplier).strip(), "quantity"].sum()) if len(a) else 0.0
    ok = current.status == "optimal" and adapt.status == "optimal"
    return {"current": current, "adapt": adapt, "units_affected": units,
            "extra_cost": adapt.total_expected_cost - current.total_expected_cost if ok else float("nan")}


def demand_change(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, pct: float, tariffs=None) -> dict:
    """Effect of all demand changing by ``pct`` (0.2 = 20% more), rounded to whole units."""
    current = optimize(demand, offers, settings, tariffs)
    changed = demand.copy()
    changed["quantity"] = (pd.to_numeric(changed["quantity"]) * (1 + pct)).round().clip(lower=1).astype(int)
    adapt = optimize(changed, offers, settings, tariffs)
    ok = current.status == "optimal" and adapt.status == "optimal"
    return {"current": current, "adapt": adapt, "demand": changed,
            "extra_cost": adapt.total_expected_cost - current.total_expected_cost if ok else float("nan")}


def vulnerability_table(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, tariff_step: float = 0.10,
                        tariffs=None) -> pd.DataFrame:
    """Rank single events by what they would cost: each supplier lost, each foreign duty raised.

    One row per event with ``extra_cost`` after re-optimizing, the same as a
    share of today's cost, and ``outcome``: ``absorbed`` (a plan still
    exists) or ``no feasible plan``. For tariff rows ``if_unchanged`` is the
    cost of not adapting. Worst events first.
    """
    current = optimize(demand, offers, settings, tariffs)
    if current.status != "optimal":
        raise ValueError(f"The current plan is not solvable ({current.status}); fix it before stress testing")
    base = current.total_expected_cost
    rows = []
    for supplier in sorted(current.allocations["supplier"].unique()):
        out = supplier_outage(demand, offers, settings, supplier, tariffs)
        feasible = out["adapt"].status == "optimal"
        rows.append({"event": f"Lose {supplier}", "type": "supplier outage", "units_affected": out["units_affected"],
                     "if_unchanged": float("nan"), "extra_cost": out["extra_cost"],
                     "outcome": "absorbed" if feasible else "no feasible plan"})
    home = str(settings.home_country).strip()
    prepared = prepare(demand, offers, settings, tariffs)[1]
    for country in sorted(c for c in prepared["country"].unique() if c != home):
        out = tariff_shock(demand, offers, settings, country, tariff_step, tariffs)
        a = current.allocations
        feasible = out["adapt"] is not None and out["adapt"].status == "optimal"
        rows.append({"event": f"Duty on {country} +{tariff_step * 100:.0f} pts", "type": "tariff",
                     "units_affected": float(a.loc[a["country"] == country, "quantity"].sum()),
                     "if_unchanged": out["exposure"], "extra_cost": out["net_impact"],
                     "outcome": "absorbed" if feasible else "no feasible plan"})
    table = pd.DataFrame(rows)
    table["extra_cost_pct"] = table["extra_cost"] / base
    table["_rank"] = table["extra_cost"].fillna(float("inf"))
    return table.sort_values("_rank", ascending=False).drop(columns="_rank").reset_index(drop=True)


def frontier(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, penalties, tariffs=None) -> pd.DataFrame:
    """Cost against delivery risk as the penalty per late unit rises.

    One row per penalty with the plan's cost before lateness
    (``cost_before_late`` = landed + fixed + financing), expected late units,
    suppliers used and domestic share. Infeasible penalties are skipped.
    """
    rows = []
    for penalty in penalties:
        plan = optimize(demand, offers, replace(settings, late_penalty_per_unit=float(penalty)), tariffs)
        if plan.status != "optimal":
            continue
        rows.append({"late_penalty_per_unit": float(penalty),
                     "cost_before_late": plan.landed_cost + plan.fixed_cost + plan.financing_cost,
                     "expected_late_units": plan.expected_late_units, "average_risk": plan.average_risk,
                     "suppliers_used": plan.suppliers_used, "domestic_share": plan.domestic_share})
    return pd.DataFrame(rows)


def sweep(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, field: str, values, tariffs=None) -> pd.DataFrame:
    """Re-plan for each value of one setting, to see what a rule costs.

    For example ``sweep(..., "min_domestic_share", [0, 0.25, 0.5, 0.75])``
    shows the price of sourcing more at home. One row per value; rows where
    no plan exists have status ``infeasible`` and empty costs.
    """
    if not hasattr(settings, field):
        raise ValueError(f"Unknown setting '{field}'")
    rows = []
    for value in values:
        plan = optimize(demand, offers, replace(settings, **{field: value}), tariffs)
        ok = plan.status == "optimal"
        rows.append({field: value, "status": plan.status,
                     "total_expected_cost": plan.total_expected_cost if ok else float("nan"),
                     "cost_before_late": plan.landed_cost + plan.fixed_cost + plan.financing_cost if ok else float("nan"),
                     "expected_late_units": plan.expected_late_units if ok else float("nan"),
                     "upfront_cash": plan.upfront_cash if ok else float("nan"),
                     "domestic_share": plan.domestic_share if ok else float("nan"),
                     "suppliers_used": plan.suppliers_used if ok else 0})
    return pd.DataFrame(rows)


_RELAXATIONS = {
    "the lead-time limit": {"deadline": None},
    "the budget": {"budget": None},
    "the upfront cash limit": {"upfront_cash_limit": None},
    "the supplier share limit": {"max_supplier_share": 1.0},
    "the minimum domestic share": {"min_domestic_share": 0.0},
    "the country share limit": {"max_country_share": 1.0},
    "the minimum number of suppliers per item": {"min_suppliers_per_sku": 1},
}


def diagnose(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings, tariffs=None) -> list[str]:
    """When no plan exists, name the rules whose removal alone would make one possible.

    Returns an empty list if the plan is already feasible, or if no single
    rule is to blame (several rules conflict together, or demand exceeds the
    capacity on offer).
    """
    if optimize(demand, offers, settings, tariffs).status == "optimal":
        return []
    fixes = []
    for label, change in _RELAXATIONS.items():
        if all(getattr(settings, k) == v for k, v in change.items()):
            continue  # this rule is not active
        if optimize(demand, offers, replace(settings, **change), tariffs).status == "optimal":
            fixes.append(label)
    return fixes
