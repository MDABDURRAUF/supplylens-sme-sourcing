from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import sme_sourcing as ss
from sme_sourcing import Settings, evaluate_plan, optimize
from sme_sourcing.cli import main


def qty(result, by="supplier"):
    return result.allocations.groupby(by)["quantity"].sum().to_dict()


@pytest.fixture
def inputs():
    demand = pd.DataFrame([{"sku": "widget", "quantity": 100}])
    offers = pd.DataFrame([
        {"sku": "widget", "supplier": "A", "unit_cost": 2, "capacity": 60, "min_order": 20, "lead_time": 5, "risk": 0.5},
        {"sku": "widget", "supplier": "B", "unit_cost": 3, "capacity": 100, "min_order": 20, "lead_time": 8, "risk": 0.1},
    ])
    return demand, offers


@pytest.fixture
def sample():
    return ss.sample()


# ---------------------------------------------- behaviour kept from the original optimizer
def test_cost_baseline_and_risk_adjusted_plan(inputs):
    demand, offers = inputs
    baseline = optimize(demand, offers)
    safer = optimize(demand, offers, Settings(late_penalty_per_unit=5))
    assert baseline.status == safer.status == "optimal"
    assert qty(baseline) == {"A": 60, "B": 40}
    assert baseline.goods_cost == pytest.approx(240) and baseline.expected_late_units == pytest.approx(34)
    assert qty(safer) == {"B": 100}
    assert safer.goods_cost == pytest.approx(300) and safer.expected_late_cost == pytest.approx(50)
    assert safer.total_expected_cost == pytest.approx(350) and safer.average_risk == pytest.approx(0.1)
    # With no duty, freight, fixed cost or financing, landed cost is the goods cost.
    assert safer.landed_cost == safer.goods_cost and safer.duty_cost == safer.freight_cost == safer.fixed_cost == 0


def test_minimum_orders_deadline_budget_and_share(inputs):
    demand, offers = inputs
    tight = offers.copy()
    tight.loc[tight.supplier == "A", ["capacity", "min_order"]] = [100, 60]
    tight.loc[tight.supplier == "B", ["capacity", "min_order", "unit_cost"]] = [80, 50, 1]
    assert qty(optimize(demand, tight)) == {"A": 100}
    assert optimize(demand, tight, Settings(max_supplier_share=0.6)).status == "infeasible"
    assert optimize(demand, offers, Settings(deadline=5)).status == "infeasible"
    assert optimize(demand, offers, Settings(budget=239)).status == "infeasible"
    assert optimize(demand, offers, Settings(budget=240)).goods_cost == pytest.approx(240)


def test_supplier_share_applies_across_skus():
    demand = pd.DataFrame([{"sku": "x", "quantity": 50}, {"sku": "y", "quantity": 50}])
    offers = pd.DataFrame([
        {"sku": sku, "supplier": supplier, "unit_cost": cost, "capacity": 50, "min_order": 0, "lead_time": 2, "risk": .1}
        for sku in ("x", "y") for supplier, cost in (("A", 1), ("B", 2))
    ])
    result = optimize(demand, offers, Settings(max_supplier_share=.6))
    assert qty(result) == {"A": 60, "B": 40} and result.goods_cost == pytest.approx(140)
    assert result.top_supplier_share == pytest.approx(0.6) and result.suppliers_used == 2


def test_validation(inputs):
    demand, offers = inputs
    with pytest.raises(ValueError, match="at most one row"):
        optimize(demand, pd.concat([offers, offers.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="risk must"):
        optimize(demand, offers.assign(risk=1.5))
    with pytest.raises(ValueError, match="whole units"):
        optimize(demand.assign(quantity=10.5), offers)
    with pytest.raises(ValueError, match="max_supplier_share"):
        optimize(demand, offers, Settings(max_supplier_share=0))
    with pytest.raises(ValueError, match="min_order cannot"):
        optimize(demand, offers.assign(min_order=1000))
    with pytest.raises(ValueError, match="finite"):
        optimize(demand, offers.assign(unit_cost=float("nan")))
    with pytest.raises(ValueError, match="deposit_pct"):
        optimize(demand, offers.assign(deposit_pct=1.2))
    with pytest.raises(ValueError, match="fixed_order_cost must be the same"):
        optimize(pd.DataFrame({"sku": ["x", "y"], "quantity": [1, 1]}),
                 pd.DataFrame({"sku": ["x", "y"], "supplier": ["A", "A"], "unit_cost": 1, "capacity": 5, "min_order": 0,
                               "lead_time": 1, "risk": 0, "fixed_order_cost": [10, 20]}))
    with pytest.raises(ValueError, match="single country"):
        optimize(pd.DataFrame({"sku": ["x", "y"], "quantity": [1, 1]}),
                 pd.DataFrame({"sku": ["x", "y"], "supplier": ["A", "A"], "country": ["US", "MX"], "unit_cost": 1,
                               "capacity": 5, "min_order": 0, "lead_time": 1, "risk": 0}))
    with pytest.raises(ValueError, match="min_suppliers_per_sku"):
        optimize(demand, offers, Settings(min_suppliers_per_sku=0))
    with pytest.raises(ValueError, match="objective"):
        optimize(demand, offers, objective="cheapest")
    with pytest.raises(ValueError, match="country, duty_rate"):
        optimize(demand, offers, tariffs=pd.DataFrame({"place": ["CN"], "rate": [0.1]}))


# ------------------------------------------------------------------ landed cost
def test_duty_and_freight_change_the_cheapest_supplier(inputs):
    demand, offers = inputs
    offers = offers.assign(country=["CN", "US"], capacity=[100, 100], freight_per_unit=[0.4, 0.1])
    assert qty(optimize(demand, offers)) == {"A": 100}  # 2.40 landed against 3.10
    taxed = optimize(demand, offers, tariffs={"CN": 0.5})  # A becomes 2 + 1 + 0.4 = 3.40
    assert qty(taxed) == {"B": 100}
    assert taxed.duty_cost == 0 and taxed.freight_cost == pytest.approx(10) and taxed.landed_cost == pytest.approx(310)
    forced = optimize(demand, offers[offers.supplier == "A"], tariffs={"CN": 0.5})
    assert forced.goods_cost == pytest.approx(200) and forced.duty_cost == pytest.approx(100)
    assert forced.freight_cost == pytest.approx(40) and forced.landed_cost == pytest.approx(340)
    assert forced.domestic_share == 0 and forced.country_shares == {"CN": 1.0}


def test_duty_rate_precedence_and_home_country(inputs):
    demand, offers = inputs
    offers = offers.assign(country=["CN", "US"], duty_rate=[0.1, 0.9])
    _, prepared = ss.prepare(demand, offers, tariffs={"CN": 0.5, "US": 0.3})
    rates = prepared.set_index("supplier")["duty_rate"]
    assert rates["A"] == 0.1  # the offer's own rate wins over the country table
    assert rates["B"] == 0.0  # the home country never pays import duty
    blank = offers.assign(duty_rate=[np.nan, np.nan])
    assert ss.prepare(demand, blank, tariffs=pd.DataFrame({"country": ["CN"], "duty_rate": [0.5]}))[1].set_index("supplier")["duty_rate"]["A"] == 0.5
    assert ss.prepare(demand, blank)[1]["duty_rate"].eq(0).all()
    # Changing the home country changes who is domestic.
    assert ss.prepare(demand, offers, Settings(home_country="CN"))[1].set_index("supplier")["duty_rate"]["A"] == 0.0


def test_fixed_order_cost_consolidates_suppliers():
    demand = pd.DataFrame({"sku": ["x", "y"], "quantity": [100, 100]})
    offers = pd.DataFrame([
        {"sku": "x", "supplier": "A", "unit_cost": 1.00, "fixed_order_cost": 0},
        {"sku": "y", "supplier": "A", "unit_cost": 1.10, "fixed_order_cost": 0},
        {"sku": "y", "supplier": "B", "unit_cost": 1.00, "fixed_order_cost": 50},
    ]).assign(capacity=100, min_order=0, lead_time=5, risk=0.0)
    plan = optimize(demand, offers)  # B saves $10 on y but costs $50 to use
    assert qty(plan) == {"A": 200} and plan.fixed_cost == 0 and plan.total_expected_cost == pytest.approx(210)
    cheap_fixed = optimize(demand, offers.assign(fixed_order_cost=[0, 0, 5]))
    assert qty(cheap_fixed) == {"A": 100, "B": 100} and cheap_fixed.fixed_cost == 5
    assert cheap_fixed.total_expected_cost == pytest.approx(205)


def test_cash_limit_and_financing_cost(inputs):
    demand, offers = inputs
    offers = offers.assign(capacity=[100, 100], deposit_pct=[1.0, 0.0], payment_terms_days=[0, 30], lead_time=[73, 73])
    free = optimize(demand, offers)
    assert qty(free) == {"A": 100} and free.upfront_cash == pytest.approx(200)
    # Supplier A needs $2 a unit up front; B needs nothing until 30 days after delivery.
    limited = optimize(demand, offers, Settings(upfront_cash_limit=100))
    assert qty(limited) == {"A": 50, "B": 50} and limited.upfront_cash == pytest.approx(100)
    assert optimize(demand, offers.assign(deposit_pct=1.0), Settings(upfront_cash_limit=100)).status == "infeasible"
    # 73 days at 50% a year costs 10% of the cash tied up; 30 days of credit on B is worth 3 x 0.5 x 30/365 a unit.
    costly = optimize(demand, offers[offers.supplier == "A"], Settings(annual_cost_of_capital=0.5))
    assert costly.financing_cost == pytest.approx(200 * 0.5 * 73 / 365)
    credit = optimize(demand, offers[offers.supplier == "B"], Settings(annual_cost_of_capital=0.5))
    assert credit.financing_cost == pytest.approx(-300 * 0.5 * 30 / 365)
    assert credit.total_expected_cost == pytest.approx(300 + credit.financing_cost)


# ---------------------------------------------------------------- sourcing rules
def test_domestic_share_country_cap_and_dual_sourcing():
    demand = pd.DataFrame({"sku": ["x"], "quantity": [100]})
    offers = pd.DataFrame({"sku": "x", "supplier": ["Home", "Asia1", "Asia2", "Mex"], "country": ["US", "CN", "CN", "MX"],
                           "unit_cost": [5.0, 1.0, 1.2, 2.0], "capacity": 100, "min_order": 0, "lead_time": 5, "risk": 0.0})
    assert qty(optimize(demand, offers)) == {"Asia1": 100}
    domestic = optimize(demand, offers, Settings(min_domestic_share=0.4))
    assert qty(domestic) == {"Home": 40, "Asia1": 60} and domestic.domestic_share == pytest.approx(0.4)
    capped = optimize(demand, offers, Settings(max_country_share=0.3))
    # At most 30 units from CN and 30 from MX; the remaining 40 must be domestic. The cap never limits the home country.
    assert qty(capped, "country") == {"CN": 30, "MX": 30, "US": 40}
    dual = optimize(demand, offers, Settings(min_suppliers_per_sku=2))
    assert qty(dual) == {"Asia1": 99, "Asia2": 1}  # a second supplier must really supply something
    assert optimize(demand, offers[offers.supplier == "Home"], Settings(min_suppliers_per_sku=2)).status == "infeasible"
    assert optimize(demand, offers[offers.country != "US"], Settings(min_domestic_share=0.1)).status == "infeasible"


# ---------------------------------------------------------------- sample and comparisons
def test_sample_plan_obeys_every_rule_and_adds_up(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    assert plan.status == "optimal"
    a = plan.allocations
    assert a.groupby("sku")["quantity"].sum().to_dict() == dict(zip(demand["sku"], demand["quantity"]))
    limits = offers.set_index(["sku", "supplier"])
    for row in a.itertuples(index=False):
        assert limits.loc[(row.sku, row.supplier), "min_order"] <= row.quantity <= limits.loc[(row.sku, row.supplier), "capacity"]
    assert plan.top_supplier_share <= settings.max_supplier_share + 1e-9
    assert plan.landed_cost == pytest.approx(plan.goods_cost + plan.duty_cost + plan.freight_cost)
    assert plan.total_expected_cost == pytest.approx(sum(plan.cost_breakdown.values()))
    assert sum(plan.country_shares.values()) == pytest.approx(1.0)
    assert evaluate_plan(demand, offers, a, settings, tariffs).violations == []


def test_lowest_price_baseline_looks_cheaper_but_costs_more(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    baseline = ss.baseline_lowest_price(demand, offers, settings, tariffs)
    assert baseline.status == "optimal"
    assert baseline.goods_cost < plan.goods_cost  # cheaper on the price list
    assert baseline.total_expected_cost > plan.total_expected_cost  # dearer in full
    # Nothing obeying the same rules can beat the optimized plan on total expected cost.
    for kw in (dict(late_penalty_per_unit=0.0), dict(annual_cost_of_capital=0.0)):
        other = optimize(demand, offers, replace(settings, **kw), tariffs)
        costed = evaluate_plan(demand, offers, other.allocations, settings, tariffs)
        assert costed.total_expected_cost >= plan.total_expected_cost - 1e-6


def test_evaluate_plan_reports_broken_rules(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    same = evaluate_plan(demand, offers, plan.allocations, settings, tariffs)
    assert same.status == "evaluated" and same.total_expected_cost == pytest.approx(plan.total_expected_cost)
    strict = evaluate_plan(demand, offers, plan.allocations, replace(settings, min_domestic_share=0.9, budget=1000, deadline=5), tariffs)
    text = " | ".join(strict.violations)
    assert "domestic share" in text and "exceeds the budget" in text and "lead-time limit" in text
    short = plan.allocations.copy()
    short.loc[0, "quantity"] -= 10
    assert any("plan supplies" in v for v in evaluate_plan(demand, offers, short, settings, tariffs).violations)
    gone = offers[offers["supplier"] != plan.allocations.loc[0, "supplier"]]
    assert any("no longer offers" in v for v in evaluate_plan(demand, gone, plan.allocations, settings, tariffs).violations)
    with pytest.raises(ValueError, match="allocations is missing"):
        evaluate_plan(demand, offers, plan.allocations.drop(columns="quantity"), settings, tariffs)


# ------------------------------------------------------------------ stress tests
def test_tariff_shock_exposure_and_adaptation(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    mx_goods = plan.allocations.loc[plan.allocations["country"] == "MX", "goods_cost"].sum()
    mx_cash = plan.allocations.loc[plan.allocations["country"] == "MX"]
    shock = ss.tariff_shock(demand, offers, settings, "MX", 0.10, tariffs)
    # Leaving the plan alone costs the new duty plus financing it over the lead time.
    extra_duty = 0.10 * mx_goods
    financing = settings.annual_cost_of_capital / 365 * (0.10 * mx_cash["goods_cost"] * mx_cash["lead_time"]).sum()
    assert shock["exposure"] == pytest.approx(extra_duty + financing)
    assert shock["stay"].violations == []
    assert 0 <= shock["net_impact"] <= shock["exposure"] + 1e-6
    assert shock["adaptation_saving"] == pytest.approx(shock["exposure"] - shock["net_impact"])
    # A duty on a country the plan does not use costs nothing; a duty cut never hurts.
    unused = next(c for c in ("CN", "VN") if c not in plan.country_shares)
    assert ss.tariff_shock(demand, offers, settings, unused, 0.10, tariffs)["exposure"] == 0
    assert ss.tariff_shock(demand, offers, settings, "CN", -0.25, tariffs)["net_impact"] <= 1e-6
    shifted = ss.with_duty_change(demand, offers, "CN", -1.0, settings, tariffs)
    assert shifted.loc[shifted["country"] == "CN", "duty_rate"].eq(0).all()  # never below zero


def test_supplier_outage_and_demand_change(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    used = plan.allocations["supplier"].iloc[0]
    out = ss.supplier_outage(demand, offers, settings, used, tariffs)
    assert out["units_affected"] > 0 and out["extra_cost"] >= -1e-6
    assert used not in set(out["adapt"].allocations["supplier"])
    unused = next(s for s in offers["supplier"].unique() if s not in set(plan.allocations["supplier"]))
    assert ss.supplier_outage(demand, offers, settings, unused, tariffs)["extra_cost"] == pytest.approx(0)
    only = offers[offers["sku"] == "SILICONE-LID"]
    lone = ss.supplier_outage(demand[demand["sku"] == "SILICONE-LID"], only[only["supplier"] != "Hanoi Precision"],
                              replace(settings, max_supplier_share=1.0, deadline=20), "Carolina Pack", tariffs)
    assert lone["adapt"].status == "infeasible" and np.isnan(lone["extra_cost"])
    more = ss.demand_change(demand, offers, settings, 0.10, tariffs)
    assert more["demand"]["quantity"].sum() == pytest.approx(demand["quantity"].sum() * 1.1) and more["extra_cost"] > 0


def test_vulnerability_table_and_frontier(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    table = ss.vulnerability_table(demand, offers, settings, tariff_step=0.10, tariffs=tariffs)
    outages = table[table["type"] == "supplier outage"]
    assert len(outages) == plan.suppliers_used and len(table[table["type"] == "tariff"]) == 3
    assert table["extra_cost"].is_monotonic_decreasing and (table["extra_cost"] >= -1e-6).all()
    tariff = table[table["type"] == "tariff"]
    assert (tariff["extra_cost"] <= tariff["if_unchanged"] + 1e-6).all()
    assert table["extra_cost_pct"].iloc[0] == pytest.approx(table["extra_cost"].iloc[0] / plan.total_expected_cost)
    with pytest.raises(ValueError, match="not solvable"):
        ss.vulnerability_table(demand, offers, replace(settings, upfront_cash_limit=1), tariffs=tariffs)

    front = ss.frontier(demand, offers, settings, [0, 1, 2, 4, 8, 16], tariffs)
    assert len(front) == 6
    assert front["expected_late_units"].is_monotonic_decreasing and front["cost_before_late"].is_monotonic_increasing


# ------------------------------------------------------------------- history
def test_risk_from_history_shrinks_small_samples():
    orders = pd.DataFrame({
        "supplier": ["Big"] * 100 + ["Tiny"] * 2,
        "promised_date": ["2025-01-10"] * 102,
        "delivery_date": ["2025-01-09"] * 80 + ["2025-01-12"] * 20 + ["2025-01-12"] * 2,
    })
    risk = ss.risk_from_history(orders, prior_strength=10).set_index("supplier")
    overall = 22 / 102
    assert risk.loc["Big", "raw_late_rate"] == 0.20 and risk.loc["Tiny", "raw_late_rate"] == 1.0
    assert risk.loc["Big", "risk"] == pytest.approx((20 + 10 * overall) / 110)
    assert risk.loc["Tiny", "risk"] == pytest.approx((2 + 10 * overall) / 12)
    assert risk.loc["Tiny", "risk"] < 0.5  # two late orders are not proof of a 100% late supplier
    assert ss.risk_from_history(orders, prior_strength=0).set_index("supplier").loc["Tiny", "risk"] == 1.0
    offers = pd.DataFrame({"sku": "x", "supplier": ["Big", "New"], "risk": [0.9, 0.3]})
    applied = ss.apply_risk(offers, risk.reset_index()).set_index("supplier")["risk"]
    assert applied["Big"] == pytest.approx(risk.loc["Big", "risk"]) and applied["New"] == 0.3
    vendor_style = orders.rename(columns={"supplier": "vendor"})
    assert len(ss.risk_from_history(vendor_style, supplier_col="vendor")) == 2
    with pytest.raises(ValueError, match="missing required columns"):
        ss.risk_from_history(orders.drop(columns="delivery_date"))


# -------------------------------------------------------------------- outputs
def test_purchase_orders_summary_and_report(sample):
    demand, offers, tariffs, settings = sample
    plan = optimize(demand, offers, settings, tariffs)
    baseline = ss.baseline_lowest_price(demand, offers, settings, tariffs)
    po = ss.purchase_orders(plan)
    assert po["quantity"].sum() == demand["quantity"].sum()
    np.testing.assert_allclose(po["deposit_due_at_order"] + po["balance_due"], po["goods_cost"])
    assert po["goods_cost"].sum() == pytest.approx(plan.goods_cost)
    sup = ss.supplier_summary(plan)
    assert sup["share_of_units"].sum() == pytest.approx(1.0)
    assert sup["total_cost"].sum() == pytest.approx(plan.landed_cost + plan.fixed_cost)
    assert sup["upfront_cash"].sum() == pytest.approx(plan.upfront_cash)

    text = ss.plan_summary(plan, baseline)
    assert f"${plan.total_expected_cost:,.0f}" in text and "quoted price alone" in text and "look -$" not in text
    assert "financing benefit from supplier credit" in text  # net-30 terms outweigh cash tied up on this plan
    assert "No plan could be produced" in ss.plan_summary(optimize(demand, offers, replace(settings, upfront_cash_limit=1), tariffs))
    html = ss.build_report(plan, baseline, ss.vulnerability_table(demand, offers, settings, tariffs=tariffs))
    for needle in ("Where the money goes", "Purchase order lines", "Stress test", "planning estimate", sup["supplier"].iloc[0]):
        assert needle in html


def test_cli(tmp_path, capsys):
    assert main(["sample", "--dir", str(tmp_path)]) == 0
    files = {n: str(tmp_path / f"{n}.csv") for n in ("demand", "offers", "tariffs")}
    report, po = tmp_path / "plan.html", tmp_path / "po.csv"
    code = main(["plan", "--demand", files["demand"], "--offers", files["offers"], "--tariffs", files["tariffs"],
                 "--late-penalty", "4", "--max-supplier-share", "0.6", "--cost-of-capital", "0.10",
                 "--report", str(report), "--po", str(po), "--stress"])
    out = capsys.readouterr().out
    assert code == 0 and "total expected cost" in out and report.stat().st_size > 4000 and len(pd.read_csv(po)) > 0
    assert main(["plan", "--demand", files["demand"], "--offers", files["offers"], "--cash-limit", "1"]) == 2
    assert main(["plan", "--demand", files["tariffs"], "--offers", files["offers"]]) == 1
    assert "error:" in capsys.readouterr().err


def test_sweep_and_diagnose(sample):
    demand, offers, tariffs, settings = sample
    table = ss.sweep(demand, offers, settings, "min_domestic_share", [0.0, 0.25, 0.5, 0.75, 1.0], tariffs)
    ok = table[table["status"] == "optimal"]
    # Requiring more domestic sourcing can never make the plan cheaper.
    assert ok["total_expected_cost"].is_monotonic_increasing and (ok["domestic_share"] >= ok["min_domestic_share"] - 1e-9).all()
    assert table.iloc[-1]["status"] == "infeasible" and np.isnan(table.iloc[-1]["total_expected_cost"])
    with pytest.raises(ValueError, match="Unknown setting"):
        ss.sweep(demand, offers, settings, "not_a_setting", [1], tariffs)

    assert ss.diagnose(demand, offers, settings, tariffs) == []
    assert ss.diagnose(demand, offers, replace(settings, upfront_cash_limit=1000), tariffs) == ["the upfront cash limit"]
    assert ss.diagnose(demand, offers, replace(settings, min_domestic_share=1.0), tariffs) == ["the minimum domestic share"]
    both = ss.diagnose(demand, offers, replace(settings, upfront_cash_limit=1000, min_domestic_share=1.0), tariffs)
    assert both == []  # two rules each block a plan, so removing one alone is not enough
