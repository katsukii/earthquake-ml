"""
data_load.py
~~~~~~~~~~~~
Utility functions for downloading (optional), loading, and cleaning the
earthquake CSV dataset used in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_csv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load earthquake data from a local CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    parse_dates : bool, default True
        Whether to parse 'time' and 'updated' columns as datetime.

    Returns
    -------
    pd.DataFrame
        Raw dataframe (before cleaning).
    """
    df = pd.read_csv(path)
    if parse_dates:
        for col in ("time", "updated"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning steps:
      * Drop rows with missing magnitude or depth
      * Filter obvious outliers (negative depths, mag <= 0)
      * Reset index

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Example filters – adjust thresholds as needed
    df = df.dropna(subset=["mag", "depth"])
    df = df[df["mag"] > 0]
    df = df[df["depth"] >= 0]

    # Add helper columns
    df["year"] = df["time"].dt.year
    df["depth_km"] = df["depth"]  # alias for clarity

    return df.reset_index(drop=True)


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: one‑liner to get a clean dataframe."""
    return clean_df(load_csv(path))