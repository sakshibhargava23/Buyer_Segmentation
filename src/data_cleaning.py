"""Step 1 – Data cleaning and property aggregation."""

from __future__ import annotations

import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats in client birth dates."""
    return pd.to_datetime(series, dayfirst=False, errors="coerce")


def clean_sale_price(series: pd.Series) -> pd.Series:
    """Convert currency strings to numeric values."""
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw_data(clients_path: str, properties_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    return clients, properties


def clean_clients(clients: pd.DataFrame) -> pd.DataFrame:
    """Handle missing attributes, normalize labels, and remove duplicates."""
    df = clients.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df = df.drop_duplicates(subset=["client_id"], keep="first")

    df["client_type"] = df["client_type"].str.strip().str.title()
    df["gender"] = df["gender"].str.strip().str.upper()
    df["country"] = df["country"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["acquisition_purpose"] = df["acquisition_purpose"].str.strip().str.title()
    df["loan_applied"] = df["loan_applied"].str.strip().str.title()
    df["referral_channel"] = df["referral_channel"].str.strip().str.title()

    df["date_of_birth"] = parse_dates(df["date_of_birth"])
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")

    df["gender"] = df["gender"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["region"] = df["region"].fillna("Unknown")
    df["referral_channel"] = df["referral_channel"].fillna("Unknown")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    return df


def clean_properties(properties: pd.DataFrame) -> pd.DataFrame:
    """Clean property transactions and aggregate buyer investment metrics."""
    df = properties.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df["sale_price"] = clean_sale_price(df["sale_price"])
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["listing_status"] = df["listing_status"].str.strip().str.title()
    df["unit_category"] = df["unit_category"].str.strip().str.title()

    sold = df[df["listing_status"] == "Sold"].copy()
    sold = sold[sold["client_ref"].notna()]

    agg = (
        sold.groupby("client_ref")
        .agg(
            num_properties=("listing_id", "count"),
            total_investment=("sale_price", "sum"),
            avg_sale_price=("sale_price", "mean"),
            avg_floor_area=("floor_area_sqft", "mean"),
            max_sale_price=("sale_price", "max"),
            unit_categories=("unit_category", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"client_ref": "client_id"})
    )

    return agg


def build_master_dataset(clients_path: str, properties_path: str) -> pd.DataFrame:
    """Merge cleaned client profiles with aggregated property investment data."""
    clients_raw, properties_raw = load_raw_data(clients_path, properties_path)
    clients = clean_clients(clients_raw)
    property_agg = clean_properties(properties_raw)

    master = clients.merge(property_agg, on="client_id", how="left")

    investment_cols = [
        "num_properties",
        "total_investment",
        "avg_sale_price",
        "avg_floor_area",
        "max_sale_price",
    ]
    for col in investment_cols:
        master[col] = master[col].fillna(0)

    master["unit_categories"] = master["unit_categories"].fillna("None")
    master["age"] = (
        (pd.Timestamp("2024-01-01") - master["date_of_birth"]).dt.days / 365.25
    ).clip(lower=18, upper=100)
    master["age"] = master["age"].fillna(master["age"].median())
    master["is_investor"] = (master["acquisition_purpose"] == "Investment").astype(int)
    master["loan_applied_flag"] = (master["loan_applied"] == "Yes").astype(int)
    master["price_per_sqft"] = master.apply(
        lambda r: r["avg_sale_price"] / r["avg_floor_area"] if r["avg_floor_area"] > 0 else 0,
        axis=1,
    )

    return master
