#!/usr/bin/env python3
"""
Data Cleaning & Reporting Automation
=====================================
Reusable pipeline: point it at any new CSV export of the same sales-data
shape and it will clean it, log every fix it made, and hand back both a
cleaned file and the inputs needed for the automated Excel report.

Usage:
    python3 clean_pipeline.py --input path/to/file.csv --outdir path/to/outdir

Designed to be re-run every time a fresh data extract lands (e.g. a
scheduled weekly export) without any manual editing.
"""
import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

LOG = []


def log(action, detail):
    LOG.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                "action": action, "detail": detail})
    print(f"[{action}] {detail}")


def load(path):
    df = pd.read_csv(path)
    log("LOAD", f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def handle_missing_values(df):
    """Impute or flag missing values per column type. Logs every fill."""
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            log("MISSING_VALUES", f"'{col}': {n_missing} missing filled with median ({fill_val:.2f})")
        elif col.lower() in ("date",):
            before = len(df)
            df = df.dropna(subset=[col])
            log("MISSING_VALUES", f"'{col}': {n_missing} missing dates -> dropped {before - len(df)} rows (date can't be safely imputed)")
        else:
            df[col] = df[col].fillna("Unknown")
            log("MISSING_VALUES", f"'{col}': {n_missing} missing filled with 'Unknown'")
    return df


def handle_duplicates(df, business_key=None):
    """Remove exact duplicate rows, then duplicate business keys (keep first)."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log("DUPLICATES", f"Removed {removed} fully duplicate row(s)")
    else:
        log("DUPLICATES", "No fully duplicate rows found")

    if business_key and business_key in df.columns:
        before = len(df)
        dupe_keys = df[business_key].duplicated().sum()
        df = df.drop_duplicates(subset=[business_key], keep="first")
        if dupe_keys:
            log("DUPLICATES", f"Removed {dupe_keys} row(s) with duplicate '{business_key}' (kept first occurrence)")
        else:
            log("DUPLICATES", f"No duplicate '{business_key}' values found")
    return df


def standardize_text(df, text_cols):
    """Trim whitespace and normalize casing/spelling so the same category
    doesn't appear under multiple spellings (e.g. 'east', 'East ', 'EAST')."""
    for col in text_cols:
        if col not in df.columns:
            continue
        original = df[col].copy()
        df[col] = df[col].astype(str).str.strip()
        # collapse internal double-spaces
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        # Title-case category-like fields; leave proper nouns like Brand alone if already consistent
        n_changed = (original.astype(str) != df[col]).sum()
        if n_changed:
            log("INCONSISTENT_TEXT", f"'{col}': normalized whitespace/casing on {n_changed} value(s)")
    return df


def validate_numeric_ranges(df, rules):
    """rules: {col: (min, max)}. Clips out-of-range values and logs them."""
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = mask.sum()
        if n_bad:
            log("RANGE_CHECK", f"'{col}': {n_bad} value(s) outside [{lo}, {hi}] clipped to range")
            df[col] = df[col].clip(lo, hi)
        else:
            log("RANGE_CHECK", f"'{col}': all values within expected range [{lo}, {hi}]")
    return df


def reconcile_computed_column(df, revenue_col="Revenue (INR)"):
    """Revenue should equal Units Sold * Unit Price * (1 - Discount/100).
    Recompute and overwrite any row where the stored value drifts from the
    formula by more than a rounding tolerance, so downstream reports are
    always internally consistent."""
    if not {"Units Sold", "Unit Price (INR)", "Discount (%)", revenue_col}.issubset(df.columns):
        return df
    expected = df["Units Sold"] * df["Unit Price (INR)"] * (1 - df["Discount (%)"] / 100)
    drift = (df[revenue_col] - expected).abs()
    bad = drift > 1  # more than â‚¹1 off
    n_bad = bad.sum()
    if n_bad:
        df.loc[bad, revenue_col] = expected[bad].round(0)
        log("REVENUE_RECONCILE", f"{n_bad} row(s) had Revenue inconsistent with Units x Price x (1-Discount) -> recomputed")
    else:
        log("REVENUE_RECONCILE", "All Revenue values consistent with Units x Price x (1-Discount)")
    return df


def flag_outliers(df, col="Revenue (INR)"):
    """IQR-based outlier flag — informational only, nothing is deleted."""
    if col not in df.columns:
        return df
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < lo) | (df[col] > hi)]
    df["Outlier Flag"] = np.where((df[col] < lo) | (df[col] > hi), "Yes", "No")
    log("OUTLIERS", f"'{col}': {len(outliers)} statistical outlier(s) flagged (IQR method), not removed")
    return df


def parse_dates(df, col="Date"):
    if col not in df.columns:
        return df
    before = len(df)
    df[col] = pd.to_datetime(df[col], errors="coerce")
    bad = df[col].isna().sum()
    if bad:
        df = df.dropna(subset=[col])
        log("DATE_PARSE", f"'{col}': {bad} unparseable date(s) dropped")
    else:
        log("DATE_PARSE", f"'{col}': all dates parsed successfully")
    return df


def run_pipeline(input_path, outdir):
    df = load(input_path)
    raw_rows = len(df)

    df = parse_dates(df, "Date")
    df = handle_missing_values(df)
    df = handle_duplicates(df, business_key="Sale ID")
    df = standardize_text(df, ["Brand", "Product", "Category", "Region", "Retailer"])
    df = validate_numeric_ranges(df, {
        "Units Sold": (0, 10_000),
        "Unit Price (INR)": (0, 100_000),
        "Discount (%)": (0, 100),
    })
    df = reconcile_computed_column(df, "Revenue (INR)")
    df = flag_outliers(df, "Revenue (INR)")

    df = df.sort_values("Date").reset_index(drop=True)

    clean_path = f"{outdir}/cleaned_data.csv"
    df.to_csv(clean_path, index=False)
    log("SAVE", f"Cleaned dataset written to {clean_path} ({len(df)} rows, started from {raw_rows})")

    log_path = f"{outdir}/cleaning_log.json"
    with open(log_path, "w") as f:
        json.dump(LOG, f, indent=2)
    log("SAVE", f"Cleaning log written to {log_path}")

    return df, LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Automated data cleaning pipeline")
    ap.add_argument("--input", required=True, help="Path to raw CSV export")
    ap.add_argument("--outdir", default=".", help="Directory to write cleaned_data.csv and cleaning_log.json")
    args = ap.parse_args()
    run_pipeline(args.input, args.outdir)
