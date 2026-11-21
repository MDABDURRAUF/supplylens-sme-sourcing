"""SupplyLens SME Sourcing Planner: landed-cost sourcing plans and stress tests for small businesses."""

from .history import apply_risk, risk_from_history
from .optimizer import Result, Settings, evaluate_plan, optimize, prepare
from .outputs import build_report, plan_summary, purchase_orders, supplier_summary, write_report
from .samples import sample
from .scenarios import (
    baseline_lowest_price,
    demand_change,
    diagnose,
    frontier,
    supplier_outage,
    sweep,
    tariff_shock,
    vulnerability_table,
    with_duty_change,
)

__version__ = "0.1.0"

__all__ = [
    "Result",
    "Settings",
    "apply_risk",
    "baseline_lowest_price",
    "build_report",
    "demand_change",
    "diagnose",
    "evaluate_plan",
    "frontier",
    "optimize",
    "plan_summary",
    "prepare",
    "purchase_orders",
    "risk_from_history",
    "sample",
    "supplier_outage",
    "supplier_summary",
    "sweep",
    "tariff_shock",
    "vulnerability_table",
    "with_duty_change",
    "write_report",
]
