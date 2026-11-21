# SupplyLens SME Sourcing Planner

**Sourcing plans for small and medium-sized US businesses: who to buy from, what it really costs once
duty, freight and cash are counted, and what happens if a tariff changes or a supplier fails.**

**[Try the live dashboard](https://supplylens-sme-sourcing.streamlit.app)** — runs in your browser on illustrative sample data, nothing to install.

Small importers face the same tariff changes, supplier failures and cash pressure as large companies, usually
without a sourcing team or optimization software. This tool gives a small business that analysis from three
CSV files and a browser.

It builds on the SupplyLens Sourcing Optimizer (integer allocation with capacity, minimum order quantities,
a lead-time limit, a budget and a supplier-share limit) and adds what decides sourcing for a smaller company.

## Why small businesses

- The U.S. Small Business Administration's Office of Advocacy defines a small business as an independent
  firm with fewer than 500 employees, and counted more than 36 million of them in its 2025 profiles
  ([SBA Office of Advocacy](https://advocacy.sba.gov/?p=30239)).
- In U.S. Census Bureau data for 2015, companies with fewer than 500 employees were 97% of identified goods
  importers but accounted for only about a third of goods trade by value
  ([Census Bureau](https://www.census.gov/newsroom/blogs/global-reach/2017/08/who_did_all_thatexp.html)).
  Most importers are small, and each one buys in volumes where minimum orders, fixed costs and cash matter.

## What it adds for a small business

| Need | What the planner does |
|---|---|
| "The cheapest quote is not the cheapest supplier" | **Landed cost**: import duty by country of origin and freight are added to every quote |
| Small orders, fixed charges | **Fixed order cost** per supplier and **minimum order quantities**, so the plan does not scatter tiny orders |
| Tight cash | **Deposits and payment terms**, a **limit on cash paid before goods arrive**, and the **cost of money** tied up in transit |
| Late deliveries hurt more when stock is thin | **Expected cost of lateness** from each supplier's late-delivery risk |
| Dependence on one supplier or country | Rules: most from one supplier, most from one foreign country, **two suppliers for every item** |
| "Made in USA" commitments | **Minimum domestic share**, and a chart of what each extra point of domestic sourcing costs |
| Tariff uncertainty | **Tariff stress test**: the cost of changing nothing, and the cost after re-planning |
| Few orders per supplier | **Risk from your own order history**, with small samples pulled toward your overall late rate |
| No analyst on staff | Plain-language summary, **purchase order lines**, a printable report, and a message naming the rule to relax when no plan exists |

## Example

On the illustrative sample (a housewares company buying four items from five suppliers in the US, Mexico, China
and Vietnam):

| | This plan | Buying on quoted price alone |
|---|---|---|
| Goods (price list) | $46,650 | $33,814 |
| Import duty | $240 | $6,602 |
| Freight | $3,610 | $6,612 |
| Fixed order costs | $1,370 | $2,150 |
| Financing | -$109 | $290 |
| Expected cost of late delivery | $5,680 | $14,304 |
| **Total expected cost** | **$57,441** | **$63,772** |

The price-list choice looks $12,836 cheaper and is estimated to cost $6,332 more (9.9%) in total. These numbers
come from made-up inputs and illustrate the method; they are not findings about any real company, and the
difference is an estimate, not a realized saving.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-sme-sourcing.git
cd supplylens-sme-sourcing
pip install -e ".[app,dev]"
streamlit run app.py
```

The dashboard opens with the sample. Enter your own numbers on **Your data**.

| View | What you get |
|---|---|
| **Plan** | Plain-language summary, who supplies what, where the money goes against buying on price alone, purchase order lines, downloads |
| **Stress test** | Every single event ranked by cost (each supplier lost, each foreign duty raised), plus interactive tariff, outage and demand tests |
| **Trade-offs** | Cost against reliability, the price of buying American, and what limiting any one supplier costs |
| **Your data** | Editable tables, CSV upload, templates, and late-risk estimates from your order history |

Command line:

```bash
sme-sourcing sample --dir my_data
sme-sourcing plan --demand my_data/demand.csv --offers my_data/offers.csv --tariffs my_data/tariffs.csv \
    --late-penalty 4 --max-supplier-share 0.6 --cost-of-capital 0.10 \
    --report plan.html --po purchase_orders.csv --stress
```

Python:

```python
import sme_sourcing as ss

demand, offers, tariffs, settings = ss.sample()
plan = ss.optimize(demand, offers, settings, tariffs)
baseline = ss.baseline_lowest_price(demand, offers, settings, tariffs)
print(ss.plan_summary(plan, baseline))

shock = ss.tariff_shock(demand, offers, settings, country="MX", added_rate=0.10, tariffs=tariffs)
print(shock["exposure"], shock["net_impact"])          # cost if unchanged, cost after re-planning

print(ss.vulnerability_table(demand, offers, settings, tariffs=tariffs))
```

## Input data

**demand.csv**: `sku`, `quantity` (whole units, one row per item).

**offers.csv**: one row per supplier per item.

| Column | Required | Meaning |
|---|---|---|
| `sku`, `supplier` | yes | Item and supplier |
| `unit_cost` | yes | Quoted price per unit |
| `capacity`, `min_order` | yes | Most and fewest units this supplier will sell of this item |
| `lead_time` | yes | Days from order to delivery |
| `risk` | yes | Chance a delivery is late, 0 to 1 |
| `country` | no | Country of origin; home country if omitted |
| `freight_per_unit` | no | Freight and handling per unit |
| `duty_rate` | no | Import duty as a share of unit cost; if blank, the country's rate in the tariff table is used |
| `deposit_pct` | no | Share of the price paid when ordering (default 1.0) |
| `payment_terms_days` | no | Days after delivery before the balance is due (default 0) |
| `fixed_order_cost` | no | Charged once per supplier used; same value on all of a supplier's rows |

**tariffs.csv**: `country`, `duty_rate`. The home country is always treated as zero.

> The duty rates in the sample are examples, not current rates. Look up the rate for your product's HTS code at
> [hts.usitc.gov](https://hts.usitc.gov/) and confirm with your customs broker.

## How it works

A mixed-integer program (SciPy `milp`) chooses whole units from each offer to minimize

    goods + import duty + freight + fixed order costs + financing + expected cost of late delivery

subject to: all demand met; each offer's capacity and minimum order; and whichever rules are set (supplier share,
domestic share, country share, suppliers per item, lead-time limit, budget, upfront cash limit).

- **Financing** per unit = cost of money × (cash paid up front × lead time − balance × payment terms) ÷ 365.
  It can be negative when supplier credit outweighs the cash tied up.
- **Cash up front** = deposits + duty + freight + fixed order costs.
- **Buying on quoted price alone** solves the same problem minimizing `unit_cost` only, then costs the result
  in full, so the two plans are compared like for like.
- **Stress tests** re-solve after the event. For a tariff change, "if you change nothing" costs today's plan at
  the new duty; "after re-planning" is the new optimum.
- **Risk from history**: `(late orders + k × overall late rate) ÷ (orders + k)`, so a supplier needs about `k`
  orders of its own before its record counts as much as the average.

## Limits

- Every figure is a planning estimate built from what you enter. Differences between plans are estimates,
  not realized savings.
- Duty is modelled as a single rate on the goods price per country or offer. Real duty depends on the HTS
  classification, valuation, trade agreements and any additional duties in force.
- Capacity and minimum order apply per supplier per item; shared supplier capacity, volume discounts and
  multiple delivery dates are not modelled.
- Late-delivery risks are treated as independent; one event disrupting several suppliers at once is covered
  only by the single-event stress tests.
- The planner does not assess supplier quality, contracts or compliance.

## Tests

```bash
pytest
```

Nineteen tests: the original optimizer's behaviour, each added cost and rule checked against hand-worked
numbers, the stress tests, risk estimation, the outputs and the command line.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): predict and explain supply chain disruptions
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
