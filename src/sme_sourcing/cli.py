"""Command line.

    sme-sourcing sample --dir my_data
    sme-sourcing plan --demand my_data/demand.csv --offers my_data/offers.csv --tariffs my_data/tariffs.csv \
        --late-penalty 4 --max-supplier-share 0.6 --report plan.html --po purchase_orders.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .optimizer import Settings, optimize
from .outputs import build_report, money, plan_summary, purchase_orders, write_report
from .samples import sample
from .scenarios import baseline_lowest_price, vulnerability_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sme-sourcing", description="Landed-cost sourcing plans for small businesses.")
    p.add_argument("--version", action="version", version=f"sme-sourcing {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write the illustrative demand, offers and tariff CSV files")
    s.add_argument("--dir", default=".", help="folder to write into")

    pl = sub.add_parser("plan", help="optimize a plan and write its outputs")
    pl.add_argument("--demand", required=True)
    pl.add_argument("--offers", required=True)
    pl.add_argument("--tariffs", help="CSV with country and duty_rate columns")
    pl.add_argument("--late-penalty", type=float, default=0.0, help="assumed cost of one late unit")
    pl.add_argument("--max-supplier-share", type=float, default=1.0)
    pl.add_argument("--min-domestic-share", type=float, default=0.0)
    pl.add_argument("--max-country-share", type=float, default=1.0)
    pl.add_argument("--min-suppliers-per-sku", type=int, default=1)
    pl.add_argument("--deadline", type=float, help="longest acceptable lead time in days")
    pl.add_argument("--budget", type=float, help="limit on landed cost plus fixed order costs")
    pl.add_argument("--cash-limit", type=float, help="limit on cash paid before the goods arrive")
    pl.add_argument("--cost-of-capital", type=float, default=0.0, help="yearly cost of money, e.g. 0.10")
    pl.add_argument("--home-country", default="US")
    pl.add_argument("--report", help="write the plan as an HTML report")
    pl.add_argument("--po", help="write purchase order lines to this CSV file")
    pl.add_argument("--stress", action="store_true", help="include a stress test of each supplier and tariff in the report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sample":
            folder = Path(args.dir)
            folder.mkdir(parents=True, exist_ok=True)
            demand, offers, tariffs, _ = sample()
            for name, frame in (("demand", demand), ("offers", offers), ("tariffs", tariffs)):
                frame.to_csv(folder / f"{name}.csv", index=False)
            print(f"Wrote demand.csv, offers.csv and tariffs.csv to {folder}. The values are illustrative, not real rates.")
            return 0

        demand, offers = pd.read_csv(args.demand), pd.read_csv(args.offers)
        tariffs = pd.read_csv(args.tariffs) if args.tariffs else None
        settings = Settings(
            deadline=args.deadline, budget=args.budget, max_supplier_share=args.max_supplier_share,
            late_penalty_per_unit=args.late_penalty, upfront_cash_limit=args.cash_limit,
            annual_cost_of_capital=args.cost_of_capital, min_domestic_share=args.min_domestic_share,
            max_country_share=args.max_country_share, min_suppliers_per_sku=args.min_suppliers_per_sku,
            home_country=args.home_country,
        )
        plan = optimize(demand, offers, settings, tariffs)
        if plan.status != "optimal":
            print(f"{plan.status}: {plan.message}", file=sys.stderr)
            return 2
        baseline = baseline_lowest_price(demand, offers, settings, tariffs)
        print(plan_summary(plan, baseline, settings.home_country))
        print()
        show = plan.allocations[["sku", "supplier", "country", "quantity", "landed_cost", "upfront_cash", "expected_late_units"]]
        print(show.to_string(index=False, formatters={"landed_cost": money, "upfront_cash": money,
                                                      "expected_late_units": "{:,.0f}".format}))
        if args.po:
            purchase_orders(plan).to_csv(args.po, index=False)
            print(f"\nWrote {args.po}")
        if args.report:
            stress = vulnerability_table(demand, offers, settings, tariffs=tariffs) if args.stress else None
            print(f"Wrote {write_report(build_report(plan, baseline, stress, home_country=settings.home_country), args.report)}")
    except (ValueError, KeyError, FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
