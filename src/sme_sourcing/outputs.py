"""What the plan produces: purchase orders, a plain-language summary and a one-page report."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .optimizer import Result


def money(x, digits: int = 0) -> str:
    return "n/a" if pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.{digits}f}"


def purchase_orders(plan: Result) -> pd.DataFrame:
    """One line per item per supplier, ready to turn into purchase orders.

    Adds the deposit due when the order is placed, the balance due and the
    payment terms. Supplier fixed order costs are listed in ``supplier_summary``.
    """
    a = plan.allocations
    if a.empty:
        return pd.DataFrame()
    po = a[["supplier", "country", "sku", "quantity", "unit_cost", "goods_cost", "lead_time", "deposit_pct",
            "payment_terms_days"]].copy()
    po["deposit_due_at_order"] = po["goods_cost"] * po["deposit_pct"]
    po["balance_due"] = po["goods_cost"] - po["deposit_due_at_order"]
    return po.sort_values(["supplier", "sku"]).reset_index(drop=True)


def supplier_summary(plan: Result) -> pd.DataFrame:
    """Totals per supplier: units, share of all units, costs, cash up front and longest lead time."""
    a = plan.allocations
    if a.empty:
        return pd.DataFrame()
    g = a.groupby(["supplier", "country"])
    out = g.agg(items=("sku", "nunique"), units=("quantity", "sum"), goods_cost=("goods_cost", "sum"),
                duty_cost=("duty_cost", "sum"), freight_cost=("freight_cost", "sum"),
                fixed_order_cost=("fixed_order_cost", "first"), upfront_cash=("upfront_cash", "sum"),
                expected_late_units=("expected_late_units", "sum"), longest_lead_time=("lead_time", "max")).reset_index()
    out["upfront_cash"] += out["fixed_order_cost"]
    out["share_of_units"] = out["units"] / out["units"].sum()
    out["total_cost"] = out["goods_cost"] + out["duty_cost"] + out["freight_cost"] + out["fixed_order_cost"]
    return out.sort_values("total_cost", ascending=False).reset_index(drop=True)


def plan_summary(plan: Result, baseline: Result | None = None, home_country: str = "US") -> str:
    """A short plain-language description of the plan and how it compares with buying on price alone."""
    if plan.status != "optimal":
        return f"No plan could be produced: {plan.message}"
    countries = sorted(plan.country_shares.items(), key=lambda kv: -kv[1])
    mix = ", ".join(f"{share:.0%} {('domestic' if c == home_country else c)}" for c, share in countries)
    financing = (f"{money(plan.financing_cost)} financing" if plan.financing_cost >= 0
                 else f"a {money(-plan.financing_cost)} financing benefit from supplier credit")
    text = (f"Buy from {plan.suppliers_used} supplier{'' if plan.suppliers_used == 1 else 's'} ({mix}) for a total expected "
            f"cost of {money(plan.total_expected_cost)}: {money(plan.goods_cost)} goods, {money(plan.duty_cost)} duty, "
            f"{money(plan.freight_cost)} freight, {money(plan.fixed_cost)} fixed order costs, {financing} "
            f"and {money(plan.expected_late_cost)} expected cost of late delivery. "
            f"{money(plan.upfront_cash)} is needed in cash before the goods arrive, and about "
            f"{plan.expected_late_units:,.0f} units ({plan.average_risk:.1%}) are expected to arrive late.")
    if baseline is not None and baseline.status == "optimal":
        diff = baseline.total_expected_cost - plan.total_expected_cost
        if diff > 0.5:
            text += (f" Choosing on quoted price alone would look {money(abs(baseline.goods_cost - plan.goods_cost))} "
                     f"{'cheaper' if baseline.goods_cost < plan.goods_cost else 'dearer'} on the price list but is estimated to cost "
                     f"{money(diff)} more once duty, freight, fixed costs, financing and lateness are counted "
                     f"({diff / baseline.total_expected_cost:.1%}).")
        else:
            text += " Choosing on quoted price alone gives the same total expected cost here."
    return text


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #2a9d8f;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.bar{display:grid;grid-template-columns:150px 1fr 90px;gap:10px;align-items:center;font-size:13px;margin:6px 0}
.track{background:#e6e9ef;border-radius:4px;height:14px} .fill{height:14px;border-radius:4px;background:#3a6ea5}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""


def _table(df: pd.DataFrame, formats: dict | None = None) -> str:
    if df is None or df.empty:
        return '<p class="note">Nothing to show.</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            if col in formats:
                cells.append(f'<td class="num">{escape(formats[col](val))}</td>')
            else:
                cells.append(f"<td>{escape(str(val))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_report(plan: Result, baseline: Result | None = None, vulnerabilities: pd.DataFrame | None = None,
                 title: str = "Sourcing plan", home_country: str = "US") -> str:
    """The plan as a single HTML page: summary, costs, purchase orders, and stress-test results if given."""
    pct = lambda x: "n/a" if pd.isna(x) else f"{x:.0%}"  # noqa: E731
    num = lambda x: f"{x:,.0f}"  # noqa: E731
    body = f'<p class="lead">{escape(plan_summary(plan, baseline, home_country))}</p>'
    if plan.status == "optimal":
        body += '<div class="kpis">' + "".join(
            f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>'
            for k, v in [("total expected cost", money(plan.total_expected_cost)), ("cash needed up front", money(plan.upfront_cash)),
                         ("suppliers used", str(plan.suppliers_used)), ("domestic share of units", pct(plan.domestic_share)),
                         ("largest supplier share", pct(plan.top_supplier_share)),
                         ("units expected late", f"{plan.expected_late_units:,.0f}")]) + "</div>"

        body += "<h2>Where the money goes</h2>"
        parts = {k: v for k, v in plan.cost_breakdown.items() if abs(v) > 0.005}
        top = max(parts.values(), default=1) or 1
        body += "".join(f'<div class="bar"><span>{escape(k)}</span><div class="track"><div class="fill" '
                        f'style="width:{max(0, v) / top:.1%}"></div></div><span>{money(v)}</span></div>' for k, v in parts.items())

        body += "<h2>Suppliers</h2>" + _table(
            supplier_summary(plan)[["supplier", "country", "items", "units", "share_of_units", "total_cost", "upfront_cash",
                                    "longest_lead_time"]],
            {"units": num, "share_of_units": pct, "total_cost": money, "upfront_cash": money, "longest_lead_time": lambda d: f"{d:.0f} days"})
        body += "<h2>Purchase order lines</h2>" + _table(
            purchase_orders(plan),
            {"quantity": num, "unit_cost": lambda v: money(v, 2), "goods_cost": money, "lead_time": lambda d: f"{d:.0f} days",
             "deposit_pct": pct, "payment_terms_days": lambda d: f"net {d:.0f}", "deposit_due_at_order": money, "balance_due": money})

        if baseline is not None and baseline.status == "optimal":
            body += "<h2>Compared with buying on quoted price alone</h2>"
            rows = [("Goods (price list)", "goods_cost"), ("Import duty", "duty_cost"), ("Freight", "freight_cost"),
                    ("Fixed order costs", "fixed_cost"), ("Financing", "financing_cost"), ("Expected late cost", "expected_late_cost"),
                    ("Total expected cost", "total_expected_cost"), ("Cash needed up front", "upfront_cash")]
            comp = pd.DataFrame({"": [r[0] for r in rows], "this plan": [getattr(plan, r[1]) for r in rows],
                                 "lowest quoted price": [getattr(baseline, r[1]) for r in rows]})
            comp["difference"] = comp["this plan"] - comp["lowest quoted price"]
            body += _table(comp, {"this plan": money, "lowest quoted price": money, "difference": money})

        if vulnerabilities is not None and len(vulnerabilities):
            body += "<h2>Stress test</h2><p class=\"note\">Extra cost of single events after re-optimizing, worst first.</p>"
            body += _table(vulnerabilities[["event", "units_affected", "if_unchanged", "extra_cost", "extra_cost_pct", "outcome"]],
                           {"units_affected": num, "if_unchanged": money, "extra_cost": money,
                            "extra_cost_pct": lambda x: "n/a" if pd.isna(x) else f"{x:+.1%}"})
    body += ('<h2>How to read this</h2><p class="caveat">Every figure is a planning estimate built from the prices, duty rates, '
             "lead times and late-delivery probabilities that were entered. The cost of a late unit and the cost of capital are "
             "assumptions. Differences between plans are estimates, not realized savings. Check duty rates for your product's "
             "HTS code and confirm quotes with suppliers before placing orders.</p>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">SupplyLens SME Sourcing Planner</p>{body}</main></body></html>')


def write_report(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
