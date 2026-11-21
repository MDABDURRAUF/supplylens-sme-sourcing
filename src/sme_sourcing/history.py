"""Estimate each supplier's late-delivery risk from past orders.

A small business rarely has many orders per supplier, so a raw late rate from
five orders is unreliable. The estimate here pulls each supplier's rate toward
the overall rate, strongly when there is little history and hardly at all when
there is a lot.
"""

from __future__ import annotations

import pandas as pd


def risk_from_history(orders: pd.DataFrame, supplier_col: str = "supplier", promised_col: str = "promised_date",
                      delivered_col: str = "delivery_date", prior_strength: float = 10.0) -> pd.DataFrame:
    """Late-delivery risk per supplier from an order history.

    ``orders`` needs one row per delivered order with a supplier, a promised
    date and a delivery date. (A SupplyLens orders table works with
    ``supplier_col="vendor"``.)

        risk = (late orders + prior_strength x overall late rate) / (orders + prior_strength)

    ``prior_strength`` is how many "average" orders each supplier is assumed
    to start with: 10 means a supplier needs about ten orders of its own
    before its record counts as much as the overall average.

    Returns ``supplier``, ``orders``, ``late_orders``, ``raw_late_rate`` and
    ``risk``, riskiest first.
    """
    missing = [c for c in (supplier_col, promised_col, delivered_col) if c not in orders.columns]
    if missing:
        raise ValueError(f"orders is missing required columns: {', '.join(missing)}")
    if prior_strength < 0:
        raise ValueError("prior_strength cannot be negative")
    df = orders[[supplier_col, promised_col, delivered_col]].copy()
    df[promised_col] = pd.to_datetime(df[promised_col], errors="coerce")
    df[delivered_col] = pd.to_datetime(df[delivered_col], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise ValueError("orders has no rows with a supplier, a promised date and a delivery date")
    df["late"] = df[delivered_col] > df[promised_col]
    overall = float(df["late"].mean())
    g = df.groupby(df[supplier_col].astype(str).str.strip())["late"]
    out = pd.DataFrame({"orders": g.size(), "late_orders": g.sum().astype(int)})
    out["raw_late_rate"] = out["late_orders"] / out["orders"]
    out["risk"] = (out["late_orders"] + prior_strength * overall) / (out["orders"] + prior_strength)
    out.index.name = "supplier"
    return out.sort_values("risk", ascending=False).reset_index()


def apply_risk(offers: pd.DataFrame, risk_table: pd.DataFrame) -> pd.DataFrame:
    """Replace ``risk`` in ``offers`` with the estimate for each supplier.

    Suppliers without history keep the risk already in ``offers``.
    """
    lookup = risk_table.set_index("supplier")["risk"]
    out = offers.copy()
    estimated = out["supplier"].astype(str).str.strip().map(lookup)
    out["risk"] = estimated.fillna(out["risk"]) if "risk" in out.columns else estimated
    return out
