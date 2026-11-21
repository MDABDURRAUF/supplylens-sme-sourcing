"""Illustrative data: a small US housewares company buying four items from five suppliers.

Every number here is made up for demonstration. In particular the duty rates
are NOT current tariff rates: look up the rate for your product's HTS code at
https://hts.usitc.gov/ before using this for a real decision.
"""

from __future__ import annotations

import pandas as pd

from .optimizer import Settings

DEMAND = pd.DataFrame(
    {"sku": ["STEEL-BOWL", "SILICONE-LID", "GIFT-BOX", "HANDLE-KIT"], "quantity": [4000, 6000, 5000, 3000]}
)

_COLUMNS = ["sku", "supplier", "country", "unit_cost", "freight_per_unit", "capacity", "min_order", "lead_time", "risk",
            "deposit_pct", "payment_terms_days", "fixed_order_cost"]
OFFERS = pd.DataFrame(
    [
        # item           supplier                 from  price  freight  cap   min   days  risk  deposit terms fixed
        ["STEEL-BOWL",   "Ohio Metalworks",       "US", 6.90,  0.25,   3000,  500,   14, 0.05, 0.0,   30,  150],
        ["STEEL-BOWL",   "Monterrey Industrial",  "MX", 5.60,  0.45,   4000, 1000,   21, 0.10, 0.3,   15,  400],
        ["STEEL-BOWL",   "Shenzhen Brightway",    "CN", 4.10,  0.85,   6000, 2000,   45, 0.22, 0.3,    0,  900],
        ["STEEL-BOWL",   "Hanoi Precision",       "VN", 4.60,  0.90,   5000, 1500,   50, 0.18, 0.3,    0,  850],
        ["SILICONE-LID", "Carolina Pack",         "US", 1.95,  0.08,   4000,  500,   10, 0.04, 0.0,   30,  120],
        ["SILICONE-LID", "Shenzhen Brightway",    "CN", 1.05,  0.20,   8000, 3000,   45, 0.22, 0.3,    0,  900],
        ["SILICONE-LID", "Hanoi Precision",       "VN", 1.20,  0.22,   6000, 2000,   50, 0.18, 0.3,    0,  850],
        ["GIFT-BOX",     "Carolina Pack",         "US", 1.10,  0.06,   6000,  500,   10, 0.04, 0.0,   30,  120],
        ["GIFT-BOX",     "Monterrey Industrial",  "MX", 0.92,  0.12,   5000, 1000,   21, 0.10, 0.3,   15,  400],
        ["GIFT-BOX",     "Shenzhen Brightway",    "CN", 0.70,  0.16,   8000, 3000,   45, 0.22, 0.3,    0,  900],
        ["HANDLE-KIT",   "Ohio Metalworks",       "US", 3.40,  0.15,   2500,  500,   14, 0.05, 0.0,   30,  150],
        ["HANDLE-KIT",   "Monterrey Industrial",  "MX", 2.85,  0.25,   3000,  500,   21, 0.10, 0.3,   15,  400],
        ["HANDLE-KIT",   "Hanoi Precision",       "VN", 2.30,  0.40,   4000, 1000,   50, 0.18, 0.3,    0,  850],
    ],
    columns=_COLUMNS,
)

# Illustrative duty rates by country of origin. Not current rates.
TARIFFS = pd.DataFrame({"country": ["US", "MX", "CN", "VN"], "duty_rate": [0.00, 0.00, 0.25, 0.10]})

SETTINGS = Settings(max_supplier_share=0.60, late_penalty_per_unit=4.0, annual_cost_of_capital=0.10)


def sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Settings]:
    """Copies of the sample demand, offers and tariff tables, and suggested settings."""
    return DEMAND.copy(), OFFERS.copy(), TARIFFS.copy(), SETTINGS
