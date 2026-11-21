"""Sourcing allocation for small and medium-sized businesses.

Extends the SupplyLens Sourcing Optimizer (integer allocation with supplier
capacity, minimum order quantities, a lead-time limit, a budget and a
supplier-share limit) with the costs and rules that decide a smaller
company's sourcing:

* **landed cost**: import duty and freight on top of the unit price;
* **fixed cost** of placing an order with a supplier;
* **cash**: deposits, payment terms, a limit on cash paid up front, and the
  cost of money tied up while goods are on the way;
* **sourcing rules**: a minimum domestic share, a cap on any one foreign
  country, and a minimum number of suppliers per item.

All figures are planning estimates supplied by the user. Risk is an input
(probability of late delivery), not a forecast produced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

REQUIRED_DEMAND = ["sku", "quantity"]
REQUIRED_OFFER = ["sku", "supplier", "unit_cost", "capacity", "min_order", "lead_time", "risk"]
# Optional offer columns and the value used when a column is absent.
OPTIONAL_OFFER = {
    "freight_per_unit": 0.0,      # freight and handling per unit, in currency
    "duty_rate": np.nan,          # import duty as a share of unit cost; blank = look up by country
    "deposit_pct": 1.0,           # share of the goods price paid when the order is placed
    "payment_terms_days": 0.0,    # days after delivery before the balance is due
    "fixed_order_cost": 0.0,      # one charge per supplier used (same value on all of a supplier's rows)
}
COST_COLUMNS = ["goods_cost", "duty_cost", "freight_cost", "financing_cost", "landed_cost", "upfront_cash",
                "expected_late_units", "expected_late_cost"]


@dataclass(frozen=True)
class Settings:
    """Rules and assumptions for one plan. Money values share the currency of the prices."""

    deadline: float | None = None            # longest acceptable lead time, in days
    budget: float | None = None              # limit on landed cost plus fixed order costs
    max_supplier_share: float = 1.0          # largest share of all units from one supplier
    late_penalty_per_unit: float = 0.0       # assumed cost of one unit arriving late
    upfront_cash_limit: float | None = None  # limit on cash paid before the goods arrive
    annual_cost_of_capital: float = 0.0      # yearly cost of money, e.g. 0.10 for a 10% credit line
    min_domestic_share: float = 0.0          # smallest share of all units from the home country
    max_country_share: float = 1.0           # largest share of all units from any one foreign country
    min_suppliers_per_sku: int = 1           # 2 = dual sourcing for every item
    home_country: str = "US"


@dataclass
class Result:
    """A plan and its cost.

    ``total_expected_cost`` = landed cost + fixed order costs + financing cost
    + expected late cost, and is what the optimizer minimizes.
    """

    status: str
    message: str
    allocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    goods_cost: float = 0.0
    duty_cost: float = 0.0
    freight_cost: float = 0.0
    fixed_cost: float = 0.0
    financing_cost: float = 0.0
    landed_cost: float = 0.0          # goods + duty + freight
    expected_late_units: float = 0.0
    expected_late_cost: float = 0.0
    total_expected_cost: float = 0.0
    upfront_cash: float = 0.0
    average_risk: float = 0.0
    suppliers_used: int = 0
    domestic_share: float = 0.0
    top_supplier_share: float = 0.0
    country_shares: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)

    @property
    def cost_breakdown(self) -> dict:
        return {"Goods": self.goods_cost, "Import duty": self.duty_cost, "Freight": self.freight_cost,
                "Fixed order costs": self.fixed_cost, "Financing": self.financing_cost,
                "Expected late cost": self.expected_late_cost}


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be a finite number at least {minimum:g}")
    return number


def _share(value: object, name: str, *, allow_zero: bool) -> float:
    share = _number(value, name)
    if share > 1 or (share == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'between 0 and 1' if allow_zero else 'greater than 0 and at most 1'}")
    return share


def _tariff_lookup(tariffs) -> dict:
    """Country -> duty rate, from a dict or a table with ``country`` and ``duty_rate`` columns."""
    if tariffs is None:
        return {}
    if isinstance(tariffs, pd.DataFrame):
        if not {"country", "duty_rate"} <= set(tariffs.columns):
            raise ValueError("tariffs needs the columns: country, duty_rate")
        tariffs = dict(zip(tariffs["country"].astype(str).str.strip(), tariffs["duty_rate"]))
    lookup = {str(k).strip(): _number(v, "tariff duty_rate") for k, v in dict(tariffs).items()}
    return lookup


def prepare(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings = Settings(), tariffs=None,
            apply_deadline: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the inputs and return ``(demand, offers)`` with every cost per unit worked out.

    Added offer columns: ``country`` (home country if not given), ``duty_rate``
    (the offer's own rate, else the country's rate in ``tariffs``, else 0; the
    home country is always 0), and per unit: ``duty_per_unit``,
    ``landed_per_unit``, ``upfront_per_unit``, ``financing_per_unit``.
    """
    if not isinstance(demand, pd.DataFrame) or not isinstance(offers, pd.DataFrame):
        raise ValueError("demand and offers must be pandas DataFrames")
    if not isinstance(settings, Settings):
        raise ValueError("settings must be a Settings instance")
    for frame, required, name in ((demand, REQUIRED_DEMAND, "demand"), (offers, REQUIRED_OFFER, "offers")):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name} has duplicate column names")

    demand = demand.loc[:, REQUIRED_DEMAND].copy()
    keep = REQUIRED_OFFER + [c for c in ["country", *OPTIONAL_OFFER] if c in offers.columns]
    offers = offers.loc[:, keep].copy()
    if demand.empty:
        raise ValueError("demand must have at least one row")
    home = str(settings.home_country).strip()
    if "country" not in offers:
        offers["country"] = home
    offers["country"] = offers["country"].fillna(home)
    for column, default in OPTIONAL_OFFER.items():
        if column not in offers:
            offers[column] = default

    for frame, columns in ((demand, ["sku"]), (offers, ["sku", "supplier", "country"])):
        for column in columns:
            if frame[column].isna().any():
                raise ValueError(f"{column} cannot be blank")
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise ValueError(f"{column} cannot be blank")
    if demand["sku"].duplicated().any():
        raise ValueError("demand must contain one row per SKU")
    if offers.duplicated(["sku", "supplier"]).any():
        raise ValueError("offers must contain at most one row per supplier and SKU")
    if not offers["sku"].isin(demand["sku"]).all():
        raise ValueError("every offer SKU must appear in demand")
    if offers.groupby("supplier")["country"].nunique().gt(1).any():
        raise ValueError("each supplier must have a single country")

    # Fill defaults for blank optional cells before the numeric checks.
    for column, default in OPTIONAL_OFFER.items():
        offers[column] = pd.to_numeric(offers[column], errors="coerce")
        if column != "duty_rate":
            offers[column] = offers[column].fillna(default)
    lookup = _tariff_lookup(tariffs)
    offers["duty_rate"] = offers["duty_rate"].fillna(offers["country"].map(lookup)).fillna(0.0)
    offers.loc[offers["country"] == home, "duty_rate"] = 0.0

    numeric = {"demand": (demand, ["quantity"]),
               "offers": (offers, ["unit_cost", "capacity", "min_order", "lead_time", "risk", *OPTIONAL_OFFER])}
    for frame, columns in numeric.values():
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column} must contain finite nonnegative numbers")
    for frame, column in ((demand, "quantity"), (offers, "capacity"), (offers, "min_order")):
        if not np.equal(frame[column], np.floor(frame[column])).all():
            raise ValueError(f"{column} must contain whole units")
    if (demand["quantity"] <= 0).any():
        raise ValueError("demand quantity must be greater than zero")
    if (offers["min_order"] > offers["capacity"]).any():
        raise ValueError("min_order cannot exceed capacity")
    if (offers["risk"] > 1).any():
        raise ValueError("risk must be between 0 and 1")
    if (offers["deposit_pct"] > 1).any():
        raise ValueError("deposit_pct must be between 0 and 1")
    if offers.groupby("supplier")["fixed_order_cost"].nunique().gt(1).any():
        raise ValueError("fixed_order_cost must be the same on every row of a supplier")

    deadline = None if settings.deadline is None else _number(settings.deadline, "deadline")
    if settings.budget is not None:
        _number(settings.budget, "budget")
    if settings.upfront_cash_limit is not None:
        _number(settings.upfront_cash_limit, "upfront_cash_limit")
    _share(settings.max_supplier_share, "max_supplier_share", allow_zero=False)
    _share(settings.max_country_share, "max_country_share", allow_zero=False)
    _share(settings.min_domestic_share, "min_domestic_share", allow_zero=True)
    _number(settings.late_penalty_per_unit, "late_penalty_per_unit")
    rate = _number(settings.annual_cost_of_capital, "annual_cost_of_capital")
    if int(settings.min_suppliers_per_sku) != settings.min_suppliers_per_sku or settings.min_suppliers_per_sku < 1:
        raise ValueError("min_suppliers_per_sku must be a whole number of at least 1")

    offers["duty_per_unit"] = offers["unit_cost"] * offers["duty_rate"]
    offers["landed_per_unit"] = offers["unit_cost"] + offers["duty_per_unit"] + offers["freight_per_unit"]
    # Paid before the goods arrive: the deposit, plus duty and freight.
    offers["upfront_per_unit"] = offers["unit_cost"] * offers["deposit_pct"] + offers["duty_per_unit"] + offers["freight_per_unit"]
    balance = offers["unit_cost"] * (1 - offers["deposit_pct"])
    # Cost of cash tied up during the lead time, less the value of credit on the balance after delivery.
    offers["financing_per_unit"] = rate / 365 * (offers["upfront_per_unit"] * offers["lead_time"] - balance * offers["payment_terms_days"])

    if apply_deadline and deadline is not None:
        offers = offers.loc[offers["lead_time"] <= deadline]
    offers = offers.loc[offers["capacity"] > 0].reset_index(drop=True)
    return demand.reset_index(drop=True), offers


def _summarize(demand: pd.DataFrame, offers: pd.DataFrame, quantities: np.ndarray, settings: Settings,
               status: str, message: str) -> Result:
    """Cost a set of quantities (one per row of the prepared ``offers``)."""
    penalty = float(settings.late_penalty_per_unit)
    total_demand = float(demand["quantity"].sum())
    alloc = offers.copy()
    alloc["quantity"] = np.asarray(quantities, dtype=float).round().astype(int)
    alloc = alloc.loc[alloc["quantity"] > 0].reset_index(drop=True)
    q = alloc["quantity"]
    alloc["goods_cost"] = q * alloc["unit_cost"]
    alloc["duty_cost"] = q * alloc["duty_per_unit"]
    alloc["freight_cost"] = q * alloc["freight_per_unit"]
    alloc["financing_cost"] = q * alloc["financing_per_unit"]
    alloc["landed_cost"] = q * alloc["landed_per_unit"]
    alloc["upfront_cash"] = q * alloc["upfront_per_unit"]
    alloc["expected_late_units"] = q * alloc["risk"]
    alloc["expected_late_cost"] = alloc["expected_late_units"] * penalty

    fixed = float(alloc.drop_duplicates("supplier")["fixed_order_cost"].sum())
    by_supplier = alloc.groupby("supplier")["quantity"].sum()
    by_country = alloc.groupby("country")["quantity"].sum()
    landed, financing, late_cost = (float(alloc[c].sum()) for c in ("landed_cost", "financing_cost", "expected_late_cost"))
    columns = ["sku", "supplier", "country", "quantity", "unit_cost", "duty_rate", "freight_per_unit", "lead_time", "risk",
               "deposit_pct", "payment_terms_days", "fixed_order_cost", *COST_COLUMNS]
    return Result(
        status=status,
        message=message,
        allocations=alloc[columns],
        goods_cost=float(alloc["goods_cost"].sum()),
        duty_cost=float(alloc["duty_cost"].sum()),
        freight_cost=float(alloc["freight_cost"].sum()),
        fixed_cost=fixed,
        financing_cost=financing,
        landed_cost=landed,
        expected_late_units=float(alloc["expected_late_units"].sum()),
        expected_late_cost=late_cost,
        total_expected_cost=landed + fixed + financing + late_cost,
        upfront_cash=float(alloc["upfront_cash"].sum()) + fixed,
        average_risk=float(alloc["expected_late_units"].sum()) / total_demand if total_demand else 0.0,
        suppliers_used=int(by_supplier.size),
        domestic_share=float(by_country.get(str(settings.home_country).strip(), 0.0)) / total_demand if total_demand else 0.0,
        top_supplier_share=float(by_supplier.max()) / total_demand if len(by_supplier) else 0.0,
        country_shares={k: float(v) / total_demand for k, v in by_country.items()},
    )


def optimize(demand: pd.DataFrame, offers: pd.DataFrame, settings: Settings = Settings(), tariffs=None,
             objective: str = "total") -> Result:
    """Find the lowest-cost plan that meets all demand and obeys every rule.

    ``objective="total"`` minimizes landed cost + fixed order costs +
    financing cost + expected late cost. ``objective="unit_price"`` minimizes
    the supplier's quoted price only, as a buyer comparing price lists would;
    the plan is still costed in full so the two can be compared.

    Returns a ``Result`` with status ``optimal``, ``infeasible`` or ``error``.
    """
    if objective not in ("total", "unit_price"):
        raise ValueError("objective must be 'total' or 'unit_price'")
    demand, offers = prepare(demand, offers, settings, tariffs)
    if offers.empty or not demand["sku"].isin(offers["sku"]).all():
        return Result("infeasible", "At least one item has no eligible supplier under the selected lead-time limit.")

    n = len(offers)
    suppliers = list(offers["supplier"].unique())
    s_index = offers["supplier"].map({s: j for j, s in enumerate(suppliers)}).to_numpy()
    S = len(suppliers)
    fixed = offers.drop_duplicates("supplier").set_index("supplier")["fixed_order_cost"].reindex(suppliers).to_numpy()
    home = str(settings.home_country).strip()
    foreign = [c for c in offers["country"].unique() if c != home]
    total_demand = float(demand["quantity"].sum())
    k = int(settings.min_suppliers_per_sku)

    # Variables: x_i units (integer), y_i offer used (binary), z_s supplier used (binary).
    nvar = 2 * n + S
    rows = []  # (coefficients {col: value}, low, high)

    for item in demand.itertuples(index=False):
        idx = np.flatnonzero(offers["sku"].to_numpy() == item.sku)
        rows.append(({int(i): 1.0 for i in idx}, item.quantity, item.quantity))
        if k > 1:
            rows.append(({int(n + i): 1.0 for i in idx}, k, np.inf))
    for i, offer in enumerate(offers.itertuples(index=False)):
        rows.append(({i: 1.0, n + i: -offer.capacity}, -np.inf, 0.0))
        rows.append(({i: 1.0, n + i: -max(offer.min_order, 1.0)}, 0.0, np.inf))  # a used offer supplies at least one unit
        rows.append(({n + i: 1.0, 2 * n + int(s_index[i]): -1.0}, -np.inf, 0.0))
    for j in range(S):
        rows.append(({int(i): 1.0 for i in np.flatnonzero(s_index == j)}, -np.inf, settings.max_supplier_share * total_demand))
    if settings.min_domestic_share > 0:
        idx = np.flatnonzero(offers["country"].to_numpy() == home)
        rows.append(({int(i): 1.0 for i in idx}, settings.min_domestic_share * total_demand, np.inf))
    if settings.max_country_share < 1:
        for country in foreign:
            idx = np.flatnonzero(offers["country"].to_numpy() == country)
            rows.append(({int(i): 1.0 for i in idx}, -np.inf, settings.max_country_share * total_demand))
    for limit, per_unit in ((settings.budget, "landed_per_unit"), (settings.upfront_cash_limit, "upfront_per_unit")):
        if limit is not None:
            coeffs = {i: float(v) for i, v in enumerate(offers[per_unit])}
            coeffs.update({2 * n + j: float(fixed[j]) for j in range(S)})
            rows.append((coeffs, -np.inf, float(limit)))

    A = lil_matrix((len(rows), nvar), dtype=float)
    low, high = np.empty(len(rows)), np.empty(len(rows))
    for r, (coeffs, lo, hi) in enumerate(rows):
        for col, value in coeffs.items():
            A[r, col] = value
        low[r], high[r] = lo, hi

    if objective == "total":
        per_unit = (offers["landed_per_unit"] + offers["financing_per_unit"]
                    + settings.late_penalty_per_unit * offers["risk"]).to_numpy()
        cost = np.r_[per_unit, np.zeros(n), fixed]
    else:
        cost = np.r_[offers["unit_cost"].to_numpy(), np.zeros(n + S)]
    bounds = Bounds(np.zeros(nvar), np.r_[offers["capacity"].to_numpy(), np.ones(n + S)])
    solved = milp(cost, integrality=np.ones(nvar), bounds=bounds, constraints=LinearConstraint(A.tocsr(), low, high))
    if solved.status != 0 or solved.x is None:
        if solved.status == 2:
            return Result("infeasible", "No plan satisfies all of the rules together.")
        return Result("error", f"The solver did not finish: {solved.message}")
    return _summarize(demand, offers, solved.x[:n], settings, "optimal", "Optimal plan found.")


def evaluate_plan(demand: pd.DataFrame, offers: pd.DataFrame, allocations: pd.DataFrame, settings: Settings = Settings(),
                  tariffs=None) -> Result:
    """Cost an existing plan under (possibly different) prices, duties and assumptions.

    ``allocations`` needs ``sku``, ``supplier`` and ``quantity``. Nothing is
    re-optimized. The result has status ``evaluated`` and lists in
    ``violations`` any rule the plan breaks under the given settings, so a
    plan can be checked against new conditions (for example a tariff change).
    """
    missing = [c for c in ("sku", "supplier", "quantity") if c not in allocations.columns]
    if missing:
        raise ValueError(f"allocations is missing required columns: {', '.join(missing)}")
    demand, offers = prepare(demand, offers, settings, tariffs, apply_deadline=False)
    plan = allocations.groupby(["sku", "supplier"], as_index=False)["quantity"].sum()
    merged = offers.merge(plan, on=["sku", "supplier"], how="left")
    unknown = plan.merge(offers[["sku", "supplier"]], on=["sku", "supplier"], how="left", indicator=True)
    unknown = unknown[unknown["_merge"] == "left_only"]
    result = _summarize(demand, offers, merged["quantity"].fillna(0).to_numpy(), settings, "evaluated", "Plan costed as given.")

    v = []
    for row in unknown.itertuples(index=False):
        v.append(f"{row.supplier} no longer offers {row.sku} ({row.quantity:,.0f} units cannot be sourced as planned)")
    got = result.allocations.groupby("sku")["quantity"].sum().reindex(demand["sku"]).fillna(0).to_numpy()
    for sku, need, have in zip(demand["sku"], demand["quantity"], got):
        if have != need:
            v.append(f"{sku}: plan supplies {have:,.0f} of {need:,.0f} units")
    a = result.allocations
    for row in a.merge(offers[["sku", "supplier", "capacity", "min_order"]], on=["sku", "supplier"]).itertuples(index=False):
        if row.quantity > row.capacity:
            v.append(f"{row.supplier} / {row.sku}: {row.quantity:,.0f} units exceeds capacity {row.capacity:,.0f}")
        if row.quantity < row.min_order:
            v.append(f"{row.supplier} / {row.sku}: {row.quantity:,.0f} units is below the minimum order {row.min_order:,.0f}")
    if settings.deadline is not None and (a["lead_time"] > settings.deadline).any():
        v.append("some lead times exceed the lead-time limit")
    eps = 1e-9
    if result.top_supplier_share > settings.max_supplier_share + eps:
        v.append(f"largest supplier share {result.top_supplier_share:.0%} exceeds {settings.max_supplier_share:.0%}")
    if result.domestic_share < settings.min_domestic_share - eps:
        v.append(f"domestic share {result.domestic_share:.0%} is below {settings.min_domestic_share:.0%}")
    home = str(settings.home_country).strip()
    for country, share in result.country_shares.items():
        if country != home and share > settings.max_country_share + eps:
            v.append(f"{country} share {share:.0%} exceeds {settings.max_country_share:.0%}")
    if settings.min_suppliers_per_sku > 1 and (a.groupby("sku")["supplier"].nunique() < settings.min_suppliers_per_sku).any():
        v.append(f"some items have fewer than {settings.min_suppliers_per_sku} suppliers")
    if settings.budget is not None and result.landed_cost + result.fixed_cost > settings.budget + 1e-6:
        v.append(f"landed cost {result.landed_cost + result.fixed_cost:,.0f} exceeds the budget {settings.budget:,.0f}")
    if settings.upfront_cash_limit is not None and result.upfront_cash > settings.upfront_cash_limit + 1e-6:
        v.append(f"upfront cash {result.upfront_cash:,.0f} exceeds the limit {settings.upfront_cash_limit:,.0f}")
    result.violations = v
    return result
