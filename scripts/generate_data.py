"""Generate John Citizen's synthetic banking data into data/bank.db.

Stdlib only, deterministic. 18 months of transactions ending today, with
patterns the Spending Analyst can actually find:

  * Sunday-night food delivery spikes, worse in "deadline weeks"
  * a gym membership charged every fortnight
  * one concert + hotel splurge in March 2026
  * dining out and delivery up ~25% in the most recent 6 months

Run:  uv run python scripts/generate_data.py
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bank.db"
CUSTOMER_ID = "CUST-001"
SEED = 20260912
MONTHS = 18

EVERYDAY, SAVINGS, CARD, CARLOAN = "ACC-EVERYDAY", "ACC-SAVINGS", "ACC-CARD", "ACC-CARLOAN"

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    suburb      TEXT NOT NULL,
    state       TEXT NOT NULL,
    occupation  TEXT NOT NULL,
    gross_annual_income REAL NOT NULL
);
CREATE TABLE accounts (
    account_id  TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_name TEXT NOT NULL,
    account_type TEXT NOT NULL,   -- transaction | savings | credit_card | loan
    balance     REAL NOT NULL,    -- negative for money owed
    interest_rate REAL,
    opened_date TEXT NOT NULL
);
CREATE TABLE transactions (
    txn_id      INTEGER PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_id  TEXT NOT NULL REFERENCES accounts(account_id),
    txn_date    TEXT NOT NULL,    -- YYYY-MM-DD
    merchant    TEXT NOT NULL,
    description TEXT NOT NULL,
    category    TEXT NOT NULL,
    amount      REAL NOT NULL     -- negative = money out, positive = money in
);
CREATE INDEX idx_txn_date ON transactions(txn_date);
CREATE INDEX idx_txn_category ON transactions(category);
CREATE INDEX idx_txn_customer ON transactions(customer_id);
"""

GROCERS = ["FreshMart", "GreenGrocer Co", "Corner Store", "FreshMart Express"]
CAFES = ["Bean There Cafe", "Morning Glory Coffee", "The Daily Grind"]
RESTAURANTS = ["Pasta Palace", "Golden Wok", "Souvlaki Street", "The Brass Tap"]
DELIVERY = ["QuickBite Delivery", "DoorDash Express", "Munch Runner"]
FUEL = ["Petro Plus", "FuelCo Parramatta"]
SHOPS = ["Wearhouse", "TechBarn", "HomeStyle Living", "Bookworm & Co"]
PUBS = ["The Brass Tap", "Riverside Hotel", "Odeon Cinemas"]
PHARMACY = ["Chemist Direct", "Parramatta Dental"]


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def build_rows(rng: random.Random, start: date, end: date) -> list[tuple]:
    rows: list[tuple] = []

    def add(d: date, merchant: str, desc: str, cat: str, amount: float, account: str = EVERYDAY) -> None:
        if start <= d <= end:
            rows.append((CUSTOMER_ID, account, d.isoformat(), merchant, desc, cat, round(amount, 2)))

    # Recency multiplier: dining/delivery drift up ~25% in the last 6 months.
    six_months_ago = end - timedelta(days=182)

    def drift(d: date) -> float:
        return 1.25 if d >= six_months_ago else 1.0

    # "Deadline weeks": six random weeks with extra delivery orders.
    total_weeks = ((end - start).days // 7) + 1
    deadline_weeks = set(rng.sample(range(total_weeks), 6))

    d = start
    while d <= end:
        wk = (d - start).days // 7
        dow = d.weekday()  # Mon=0

        # Salary, fortnightly on Thursdays.
        if dow == 3 and wk % 2 == 0:
            add(d, "Nimbus Technology Pty Ltd", "Salary", "income", 3040.50)
            add(d, "Internal Transfer", "Transfer to savings", "savings_transfer", -1200.00)
            add(d, "Internal Transfer", "Transfer from everyday", "savings_transfer", 1200.00, SAVINGS)
            add(d, "SwiftFit Gym", "Gym membership", "fitness", -22.00)

        # Rent, weekly on Mondays.
        if dow == 0:
            add(d, "Parramatta Property Group", "Rent", "rent", -620.00)

        # Groceries, 1-3 times a week.
        if rng.random() < 0.42:
            add(d, rng.choice(GROCERS), "Groceries", "groceries", -rng.uniform(38, 165))

        # Public transport on weekdays.
        if dow < 5 and rng.random() < 0.82:
            add(d, "CityLink Transit", "Opal top-up" if rng.random() < 0.25 else "Travel", "transport", -rng.uniform(6.4, 11.2))

        # Dining out: cafes on weekday mornings, restaurants on weekends.
        if dow < 5 and rng.random() < 0.38:
            add(d, rng.choice(CAFES), "Coffee", "dining_out", -rng.uniform(4.5, 12.0) * drift(d))
        if dow >= 4 and rng.random() < 0.45:
            add(d, rng.choice(RESTAURANTS), "Dinner", "dining_out", -rng.uniform(38, 128) * drift(d), CARD)

        # Food delivery: heavy on Sunday nights, heavier in deadline weeks.
        p_delivery = 0.62 if dow == 6 else 0.14
        if wk in deadline_weeks:
            p_delivery += 0.35
        if rng.random() < p_delivery:
            add(d, rng.choice(DELIVERY), "Food delivery", "food_delivery", -rng.uniform(26, 68) * drift(d))

        # Fuel roughly fortnightly.
        if rng.random() < 0.07:
            add(d, rng.choice(FUEL), "Fuel", "transport", -rng.uniform(58, 96))

        # Shopping, entertainment, health.
        if rng.random() < 0.10:
            add(d, rng.choice(SHOPS), "Purchase", "shopping", -rng.uniform(28, 310), CARD)
        if dow >= 4 and rng.random() < 0.18:
            add(d, rng.choice(PUBS), "Night out", "entertainment", -rng.uniform(22, 95), CARD)
        if rng.random() < 0.04:
            add(d, rng.choice(PHARMACY), "Health", "health", -rng.uniform(18, 220))

        # Monthly bills.
        if d.day == 3:
            add(d, "NetSpeed Internet", "Internet plan", "utilities", -79.00)
            add(d, "Streamly", "Streaming subscription", "subscriptions", -16.99)
        if d.day == 11:
            add(d, "TelcoOne", "Mobile plan", "utilities", -45.00)
            add(d, "TuneBox Music", "Music subscription", "subscriptions", -12.99)
        if d.day == 17:
            add(d, "Harbour Motors Finance", "Car loan repayment", "loan_repayment", -612.00)
        if d.day == 24:
            add(d, "Brightline Bank", "Credit card payment", "credit_card_payment", -450.00)
            add(d, "Brightline Bank", "Payment received", "credit_card_payment", 450.00, CARD)

        # Quarterly electricity.
        if d.day == 8 and d.month in (2, 5, 8, 11):
            add(d, "PowerGrid Energy", "Electricity bill", "utilities", -rng.uniform(278, 430))

        # Savings interest, monthly.
        if d.day == 1:
            add(d, "Brightline Bank", "Interest paid", "interest", rng.uniform(240, 305), SAVINGS)

        d += timedelta(days=1)

    # The splurge.
    add(date(2026, 3, 14), "Stadium Live Tickets", "Concert tickets x2", "entertainment", -482.00, CARD)
    add(date(2026, 3, 14), "Harbour View Hotel", "1 night stay", "travel", -389.00, CARD)
    add(date(2026, 3, 14), "The Brass Tap", "Pre-show dinner", "dining_out", -156.40, CARD)

    return rows


def main() -> None:
    rng = random.Random(SEED)
    end = date.today()
    start = end - timedelta(days=MONTHS * 30 + 15)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    con.execute(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
        (CUSTOMER_ID, "John Citizen", "1994-04-18", "Parramatta", "NSW", "IT project manager", 105000.0),
    )
    con.executemany(
        "INSERT INTO accounts VALUES (?,?,?,?,?,?,?)",
        [
            (EVERYDAY, CUSTOMER_ID, "Everyday Access", "transaction", 4312.80, 0.0, "2016-02-11"),
            (SAVINGS, CUSTOMER_ID, "Bright Saver", "savings", 78240.55, 4.85, "2016-02-11"),
            (CARD, CUSTOMER_ID, "Brightline Low Rate Card", "credit_card", -3214.60, 13.99, "2019-07-03"),
            (CARLOAN, CUSTOMER_ID, "Car Loan", "loan", -11480.00, 7.45, "2023-09-20"),
        ],
    )

    rows = build_rows(rng, start, end)
    con.executemany(
        "INSERT INTO transactions (customer_id, account_id, txn_date, merchant, description, category, amount)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()

    n, lo, hi = con.execute("SELECT COUNT(*), MIN(txn_date), MAX(txn_date) FROM transactions").fetchone()
    print(f"{DB_PATH}: {n} transactions, {lo} → {hi}")
    print("\nspend by category (last 12 months):")
    for cat, total, cnt in con.execute(
        "SELECT category, ROUND(SUM(-amount),2), COUNT(*) FROM transactions"
        " WHERE amount < 0 AND txn_date >= date('now','-12 months')"
        " GROUP BY category ORDER BY SUM(-amount) DESC"
    ):
        print(f"  {cat:<22} ${total:>10,.2f}  ({cnt} txns)")
    con.close()


if __name__ == "__main__":
    main()
