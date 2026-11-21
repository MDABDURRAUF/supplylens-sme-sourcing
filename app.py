"""SupplyLens SME Sourcing Planner dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sme_sourcing as ss
from sme_sourcing.outputs import money

st.set_page_config(page_title="SME Sourcing Planner", layout="wide")

RED, AMBER, GREEN, BLUE, GREY = "#d1495b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")
ERRORS = (ValueError, TypeError, KeyError)

# ---------------------------------------------------------------------- data
if "demand" not in st.session_state:
    d, o, t, _ = ss.sample()
    st.session_state.update(demand=d, offers=o, tariffs=t, data_version=0, source="sample")
demand, offers, tariffs = st.session_state["demand"], st.session_state["offers"], st.session_state["tariffs"]


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


@st.cache_data(max_entries=64, show_spinner=False)
def solve(demand, offers, tariffs, settings, objective="total"):
    return ss.optimize(demand, offers, settings, tariffs, objective=objective)


@st.cache_data(max_entries=16, show_spinner=False)
def stress(demand, offers, tariffs, settings, step):
    return ss.vulnerability_table(demand, offers, settings, tariff_step=step, tariffs=tariffs)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("SME Sourcing Planner")
st.sidebar.caption("Landed-cost sourcing plans for small businesses")
page = st.sidebar.radio("View", ["Plan", "Stress test", "Trade-offs", "Your data", "How it works"])
if st.session_state["source"] == "sample":
    st.sidebar.info("Using illustrative sample data. Duty rates are examples, not current rates. Enter your own on **Your data**.")

st.sidebar.divider()
st.sidebar.markdown("**Costs and assumptions**")
penalty = st.sidebar.number_input("Cost of one unit arriving late ($)", 0.0, 1000.0, 4.0, 0.5,
                                  help="Your estimate of lost sales, expediting or penalties per late unit.")
capital = st.sidebar.slider("Cost of money (yearly)", 0, 30, 10, 1, format="%d%%",
                            help="Interest on your credit line, or what your cash would otherwise earn.") / 100
st.sidebar.markdown("**Sourcing rules**")
max_share = st.sidebar.slider("Most from any one supplier", 10, 100, 60, 5, format="%d%%") / 100
min_dom = st.sidebar.slider("At least this much made in the US", 0, 100, 0, 5, format="%d%%") / 100
max_country = st.sidebar.slider("Most from any one foreign country", 10, 100, 100, 5, format="%d%%") / 100
dual = st.sidebar.checkbox("At least two suppliers for every item")
st.sidebar.markdown("**Limits**")
cash_on = st.sidebar.checkbox("Limit cash paid before goods arrive")
cash = st.sidebar.number_input("Cash limit ($)", 0.0, 1e9, 12000.0, 500.0, disabled=not cash_on)
lead_on = st.sidebar.checkbox("Limit lead time")
lead = st.sidebar.number_input("Longest lead time (days)", 0.0, 365.0, 30.0, 1.0, disabled=not lead_on)
budget_on = st.sidebar.checkbox("Limit total landed cost")
budget = st.sidebar.number_input("Budget ($)", 0.0, 1e9, 60000.0, 1000.0, disabled=not budget_on)

settings = ss.Settings(
    deadline=lead if lead_on else None, budget=budget if budget_on else None, max_supplier_share=max_share,
    late_penalty_per_unit=penalty, upfront_cash_limit=cash if cash_on else None, annual_cost_of_capital=capital,
    min_domestic_share=min_dom, max_country_share=max_country, min_suppliers_per_sku=2 if dual else 1,
)

# The plan is needed on most views; solve it once here.
plan = baseline = problem = None
try:
    plan = solve(demand, offers, tariffs, settings)
    baseline = solve(demand, offers, tariffs, settings, "unit_price")
except ERRORS as exc:
    problem = str(exc)


def need_plan():
    """Stop the page with guidance when there is no plan to show."""
    if problem:
        st.error(f"Please check your data: {problem}")
        st.caption("Open **Your data** to fix it, or reset to the sample.")
        st.stop()
    if plan.status != "optimal":
        st.warning(f"No plan satisfies every rule. {plan.message}")
        fixes = ss.diagnose(demand, offers, settings, tariffs)
        if fixes:
            st.markdown("A plan becomes possible if you relax **any one** of these:")
            for f in fixes:
                st.markdown(f"- {f}")
        else:
            st.markdown("No single rule is to blame: either several rules conflict together, or suppliers cannot cover "
                        "demand. Loosen the rules in the sidebar one at a time, or add supplier capacity on **Your data**.")
        st.stop()


# ====================================================================== PLAN
if page == "Plan":
    st.title("Your sourcing plan")
    need_plan()
    st.success(ss.plan_summary(plan, baseline))

    k = st.columns(5)
    k[0].metric("Total expected cost", money(plan.total_expected_cost))
    k[1].metric("Cash needed up front", money(plan.upfront_cash))
    k[2].metric("Made in the US", f"{plan.domestic_share:.0%}", f"{plan.suppliers_used} suppliers", delta_color="off")
    k[3].metric("Units expected late", f"{plan.expected_late_units:,.0f}", f"{plan.average_risk:.1%} of units", delta_color="off")
    if baseline.status == "optimal":
        diff = baseline.total_expected_cost - plan.total_expected_cost
        k[4].metric("Against buying on price alone", money(diff), "estimated lower total cost" if diff > 0 else "no difference",
                    delta_color="off", help="Both plans obey the same rules. An estimate, not a realized saving.")

    left, right = st.columns(2)
    with left:
        st.subheader("Who supplies what")
        fig = px.bar(plan.allocations, x="quantity", y="sku", color="supplier", orientation="h", text="quantity",
                     hover_data={"country": True, "lead_time": True, "landed_cost": ":,.0f"},
                     labels={"quantity": "Units", "sku": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        show(fig, 340)
    with right:
        st.subheader("Where the money goes")
        if baseline.status == "optimal":
            parts = pd.DataFrame({"cost": list(plan.cost_breakdown) * 2,
                                  "amount": list(plan.cost_breakdown.values()) + list(baseline.cost_breakdown.values()),
                                  "plan": ["This plan"] * 6 + ["Lowest quoted price"] * 6})
            fig = px.bar(parts, x="plan", y="amount", color="cost", labels={"amount": "Cost ($)", "plan": ""},
                         color_discrete_sequence=[BLUE, RED, AMBER, GREY, "#7b6fb0", "#c98b8b"])
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            show(fig, 340)
    st.caption("Buying on quoted price alone picks the cheapest price list. The stacked bars show what duty, freight, "
               "fixed order costs, financing and expected lateness add to each plan.")

    st.subheader("Suppliers")
    sup = ss.supplier_summary(plan)
    st.dataframe(sup[["supplier", "country", "items", "units", "share_of_units", "total_cost", "upfront_cash",
                      "expected_late_units", "longest_lead_time"]], hide_index=True,
                 column_config={"share_of_units": PCT, "total_cost": USD, "upfront_cash": USD,
                                "expected_late_units": st.column_config.NumberColumn("Units expected late", format="%.0f"),
                                "longest_lead_time": st.column_config.NumberColumn("Longest lead time (days)", format="%.0f")})

    st.subheader("Purchase order lines")
    po = ss.purchase_orders(plan)
    st.dataframe(po, hide_index=True, column_config={"unit_cost": USD, "goods_cost": USD, "deposit_pct": PCT,
                                                     "deposit_due_at_order": USD, "balance_due": USD})

    st.subheader("Take it with you")
    d = st.columns(3)
    d[0].download_button("Purchase order lines (CSV)", po.to_csv(index=False), "purchase_orders.csv", "text/csv", width="stretch")
    d[1].download_button("Supplier summary (CSV)", sup.to_csv(index=False), "supplier_summary.csv", "text/csv", width="stretch")
    try:
        table = stress(demand, offers, tariffs, settings, 0.10)
    except ERRORS:
        table = None
    d[2].download_button("Full plan report (HTML)", ss.build_report(plan, baseline, table), "sourcing_plan.html", "text/html",
                         width="stretch", type="primary")

# =============================================================== STRESS TEST
elif page == "Stress test":
    st.title("What could go wrong")
    need_plan()
    st.caption("Each test compares today's plan with the best plan after the event. Figures are estimates under your assumptions.")
    step = st.slider("Tariff increase to test (percentage points)", 5, 50, 10, 5) / 100
    table = stress(demand, offers, tariffs, settings, step)
    worst = table.iloc[0]
    blocked = table[table["outcome"] == "no feasible plan"]
    if len(blocked):
        st.error(f"**{len(blocked)} event(s) leave no workable plan**: {', '.join(blocked['event'])}. "
                 "You depend on these with no alternative under your current rules.")
    else:
        st.info(f"Every single event can be absorbed. The most expensive is **{worst['event']}**, adding "
                f"{money(worst['extra_cost'])} ({worst['extra_cost_pct']:+.1%}) after re-planning.")
    plot = table.assign(extra=table["extra_cost"].fillna(table["extra_cost"].max() if table["extra_cost"].notna().any() else 0))
    fig = px.bar(plot, x="extra", y="event", color="outcome", orientation="h",
                 color_discrete_map={"absorbed": BLUE, "no feasible plan": RED},
                 labels={"extra": "Extra cost after re-planning ($)", "event": ""})
    fig.update_yaxes(categoryorder="total ascending")
    show(fig, max(260, 42 * len(table) + 80))
    st.dataframe(table, hide_index=True, column_config={
        "units_affected": st.column_config.NumberColumn(format="%.0f"),
        "if_unchanged": st.column_config.NumberColumn("Extra cost if you change nothing", format="dollar"),
        "extra_cost": st.column_config.NumberColumn("Extra cost after re-planning", format="dollar"),
        "extra_cost_pct": st.column_config.NumberColumn("As share of today's cost", format="percent")})

    st.divider()
    tab_tariff, tab_outage, tab_demand = st.tabs(["A tariff changes", "A supplier is lost", "Demand changes"])
    with tab_tariff:
        prepared = ss.prepare(demand, offers, settings, tariffs)[1]
        foreign = sorted(c for c in prepared["country"].unique() if c != settings.home_country)
        if not foreign:
            st.info("All offers are domestic, so tariffs do not affect this plan.")
        else:
            c1, c2 = st.columns(2)
            country = c1.selectbox("Country", foreign)
            pts = c2.slider("Change in duty (percentage points)", -30, 60, 15, 5)
            shock = ss.tariff_shock(demand, offers, settings, country, pts / 100, tariffs)
            m = st.columns(3)
            m[0].metric("If you change nothing", money(shock["exposure"]), help="Today's plan costed at the new duty")
            m[1].metric("After re-planning", money(shock["net_impact"]))
            m[2].metric("Avoided by re-planning", money(shock["adaptation_saving"]))
            if shock["adapt"] is not None and shock["adapt"].status == "optimal":
                before = shock["current"].allocations.groupby("supplier")["quantity"].sum().rename("today")
                after = shock["adapt"].allocations.groupby("supplier")["quantity"].sum().rename("after the change")
                moves = pd.concat([before, after], axis=1).fillna(0).reset_index().melt(id_vars="supplier", var_name="plan", value_name="units")
                show(px.bar(moves, x="supplier", y="units", color="plan", barmode="group",
                            color_discrete_map={"today": GREY, "after the change": BLUE}, labels={"supplier": "", "units": "Units"}), 300)
                if before.reindex(after.index).fillna(0).equals(after) and len(before) == len(after):
                    st.caption("The best plan does not change: at this duty it is still cheapest to keep the same suppliers.")
    with tab_outage:
        supplier = st.selectbox("Supplier", sorted(offers["supplier"].astype(str).unique()))
        out = ss.supplier_outage(demand, offers, settings, supplier, tariffs)
        if out["adapt"].status != "optimal":
            st.error(f"Without **{supplier}** no plan satisfies your rules. {out['units_affected']:,.0f} units have no alternative source.")
        elif out["units_affected"] == 0:
            st.success(f"Today's plan does not buy from **{supplier}**, so losing it costs nothing.")
        else:
            m = st.columns(3)
            m[0].metric("Units to re-source", f"{out['units_affected']:,.0f}")
            m[1].metric("Extra cost", money(out["extra_cost"]))
            m[2].metric("Units expected late", f"{out['adapt'].expected_late_units:,.0f}",
                        f"{out['adapt'].expected_late_units - plan.expected_late_units:+,.0f} vs today", delta_color="inverse")
            st.dataframe(ss.supplier_summary(out["adapt"])[["supplier", "country", "units", "share_of_units", "total_cost"]],
                         hide_index=True, column_config={"share_of_units": PCT, "total_cost": USD})
    with tab_demand:
        pct = st.slider("Change in demand for every item", -50, 100, 20, 5, format="%d%%")
        ch = ss.demand_change(demand, offers, settings, pct / 100, tariffs)
        if ch["adapt"].status != "optimal":
            st.error("Suppliers cannot cover this demand under your rules. Add capacity or loosen a rule.")
        else:
            m = st.columns(3)
            m[0].metric("Change in total cost", money(ch["extra_cost"]))
            m[1].metric("Cash needed up front", money(ch["adapt"].upfront_cash), money(ch["adapt"].upfront_cash - plan.upfront_cash),
                        delta_color="inverse")
            m[2].metric("Suppliers used", ch["adapt"].suppliers_used, ch["adapt"].suppliers_used - plan.suppliers_used, delta_color="off")

# ================================================================ TRADE-OFFS
elif page == "Trade-offs":
    st.title("What each choice costs")
    need_plan()
    left, right = st.columns(2)
    with left:
        st.subheader("Cost against reliability")
        grid = sorted({0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, float(penalty)})
        front = ss.frontier(demand, offers, settings, grid, tariffs)
        fig = px.line(front, x="expected_late_units", y="cost_before_late", markers=True, hover_data=["late_penalty_per_unit", "suppliers_used"],
                      labels={"expected_late_units": "Units expected late", "cost_before_late": "Cost before lateness ($)"})
        fig.update_traces(line_color=BLUE)
        here = front[front["late_penalty_per_unit"] == float(penalty)]
        if len(here):
            fig.add_scatter(x=here["expected_late_units"], y=here["cost_before_late"], mode="markers", name="your setting",
                            marker=dict(color=RED, size=14, symbol="star"))
        show(fig, 360)
        st.caption("Each point is the best plan for a different cost of lateness. Moving left buys reliability; the star is your setting.")
    with right:
        st.subheader("The price of buying American")
        shares = [i / 20 for i in range(21)]
        dom = ss.sweep(demand, offers, settings, "min_domestic_share", shares, tariffs)
        ok = dom[dom["status"] == "optimal"]
        fig = px.line(ok, x="min_domestic_share", y="total_expected_cost", markers=True,
                      labels={"min_domestic_share": "Minimum share made in the US", "total_expected_cost": "Total expected cost ($)"})
        fig.update_traces(line_color=GREEN)
        fig.update_xaxes(tickformat=".0%")
        show(fig, 360)
        most = ok["min_domestic_share"].max() if len(ok) else 0
        extra = ok["total_expected_cost"].iloc[-1] - ok["total_expected_cost"].iloc[0] if len(ok) > 1 else 0
        st.caption(f"Domestic suppliers can cover at most {most:.0%} of units under your rules. Going from the cheapest mix to "
                   f"{most:.0%} domestic adds an estimated {money(extra)}.")

    st.subheader("How much one supplier should carry")
    conc = ss.sweep(demand, offers, settings, "max_supplier_share", [i / 20 for i in range(4, 21)], tariffs)
    ok = conc[conc["status"] == "optimal"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ok, x="max_supplier_share", y="total_expected_cost", markers=True,
                      labels={"max_supplier_share": "Most from any one supplier", "total_expected_cost": "Total expected cost ($)"})
        fig.update_traces(line_color=BLUE)
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
    with c2:
        fig = px.line(ok, x="max_supplier_share", y="upfront_cash", markers=True,
                      labels={"max_supplier_share": "Most from any one supplier", "upfront_cash": "Cash needed up front ($)"})
        fig.update_traces(line_color=AMBER)
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
    if len(ok) < len(conc):
        st.caption(f"Below {ok['max_supplier_share'].min():.0%} per supplier there are not enough suppliers to cover demand.")
    st.caption("Spreading orders across more suppliers lowers dependence on any one of them and usually costs more. "
               "These curves show how much, for your data.")

# ================================================================= YOUR DATA
elif page == "Your data":
    st.title("Your data")
    st.caption("Edit the tables, or upload CSV files. Nothing is saved after you close the page.")
    if problem:
        st.error(f"Current data problem: {problem}")
    v = st.session_state["data_version"]

    def read(upload):
        try:
            return pd.read_csv(BytesIO(upload.getvalue()))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read {upload.name}: {exc}")
            return None

    c = st.columns(3)
    ups = {"demand": c[0].file_uploader("Demand CSV", type="csv", key=f"up_d{v}"),
           "offers": c[1].file_uploader("Supplier offers CSV", type="csv", key=f"up_o{v}"),
           "tariffs": c[2].file_uploader("Duty rates CSV", type="csv", key=f"up_t{v}")}
    loaded = {k: read(u) for k, u in ups.items() if u is not None}
    loaded = {k: f for k, f in loaded.items() if f is not None}
    b = st.columns(4)
    if loaded and b[0].button(f"Use uploaded file(s): {', '.join(loaded)}", type="primary"):
        st.session_state.update(loaded, data_version=v + 1, source="yours")
        st.rerun()
    if b[1].button("Reset to sample data"):
        d0, o0, t0, _ = ss.sample()
        st.session_state.update(demand=d0, offers=o0, tariffs=t0, data_version=v + 1, source="sample")
        st.rerun()

    st.subheader("What you need to buy")
    new_demand = st.data_editor(demand, num_rows="dynamic", key=f"ed_d{v}", hide_index=True)
    st.subheader("What suppliers offer")
    st.caption("One row per supplier per item. `risk` is the chance a delivery is late (0 to 1). `deposit_pct` is the share paid "
               "when ordering; `payment_terms_days` is how long after delivery the rest is due. `fixed_order_cost` is charged once "
               "per supplier and must be the same on all of that supplier's rows. Leave `duty_rate` out to use the country table below.")
    new_offers = st.data_editor(offers, num_rows="dynamic", key=f"ed_o{v}", hide_index=True)
    st.subheader("Import duty by country")
    st.caption("Share of the goods price charged as import duty. Look up your product's rate by HTS code at hts.usitc.gov. "
               "The home country is always treated as zero.")
    new_tariffs = st.data_editor(tariffs, num_rows="dynamic", key=f"ed_t{v}", hide_index=True)
    if st.button("Apply edits", type="primary"):
        st.session_state.update(demand=new_demand, offers=new_offers, tariffs=new_tariffs, source="yours")
        st.rerun()

    st.subheader("Templates")
    d0, o0, t0, _ = ss.sample()
    t = st.columns(3)
    t[0].download_button("demand.csv", d0.to_csv(index=False), "demand.csv", "text/csv", width="stretch")
    t[1].download_button("offers.csv", o0.to_csv(index=False), "offers.csv", "text/csv", width="stretch")
    t[2].download_button("tariffs.csv", t0.to_csv(index=False), "tariffs.csv", "text/csv", width="stretch")

    st.subheader("Estimate late-delivery risk from your own order history")
    st.caption("Upload past orders with the columns `supplier`, `promised_date` and `delivery_date`. Suppliers with few orders "
               "are pulled toward your overall late rate, so two bad deliveries do not brand a supplier as always late.")
    hist_up = st.file_uploader("Order history CSV", type="csv", key=f"up_h{v}")
    if hist_up is not None:
        hist = read(hist_up)
        if hist is not None:
            try:
                strength = st.slider("How many orders before a supplier's own record counts as much as the average", 0, 50, 10)
                risk = ss.risk_from_history(hist, prior_strength=strength)
            except ERRORS as exc:
                st.error(str(exc))
            else:
                st.dataframe(risk, hide_index=True, column_config={"raw_late_rate": PCT, "risk": PCT})
                matched = set(risk["supplier"]) & set(offers["supplier"].astype(str).str.strip())
                if st.button(f"Use these estimates for {len(matched)} matching supplier(s)", disabled=not matched):
                    st.session_state.update(offers=ss.apply_risk(offers, risk), data_version=v + 1, source="yours")
                    st.rerun()

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.markdown(
        """
**What it does.** You say what you need to buy and what each supplier offers. The planner finds the split across
suppliers with the lowest *total* expected cost that meets all your demand and obeys your rules.

**Total expected cost** is

    goods + import duty + freight + fixed order costs + financing + expected cost of late delivery

- *Import duty* is the goods price times the duty rate for the supplier's country (zero for domestic suppliers).
- *Fixed order costs* are charged once for each supplier you use, which favours fewer suppliers.
- *Financing* is the cost of cash tied up while goods are on the way, less the value of supplier credit after delivery.
- *Expected cost of late delivery* is units times the chance of lateness times what a late unit costs you.

**Rules you can set.** Most from any one supplier; at least a share made in the US; most from any one foreign country;
two suppliers for every item; a lead-time limit; a limit on cash paid before goods arrive; a budget.

**Compared with buying on price alone.** The comparison plan obeys the same rules but chooses on the quoted price
only, as you would from a price list. Both are then costed the same way.

**Stress tests** re-plan after an event (a tariff change, a lost supplier, a change in demand) and report the extra
cost, and what it would cost to change nothing.

**What it is not.** Every number is an estimate built from what you enter. Duty rates in the sample are examples, not
current rates. The planner does not check supplier quality, contracts or compliance, and differences between plans
are estimates, not realized savings. Confirm quotes and duty rates before ordering.
"""
    )
