"""
eda.py
~~~~~~
Reusable plotting functions for exploratory data analysis (EDA).
All plots are saved to the given output directory and, optionally,
returned as Figure objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _setup_output_dir(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_mag_hist(df: pd.DataFrame, out_dir: str | Path, binwidth: float = 0.2):
    """Histogram of earthquake magnitudes."""
    out_path = _setup_output_dir(out_dir) / "01_mag_hist.png"

    plt.figure()
    sns.histplot(data=df, x="mag", binwidth=binwidth)
    plt.title("Distribution of Earthquake Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_depth_hist(df: pd.DataFrame, out_dir: str | Path, binwidth: int = 10):
    """Histogram of earthquake depths."""
    out_path = _setup_output_dir(out_dir) / "02_depth_hist.png"

    plt.figure()
    sns.histplot(data=df, x="depth_km", binwidth=binwidth)
    plt.title("Distribution of Earthquake Depths")
    plt.xlabel("Depth (km)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_yearly_trend(df: pd.DataFrame, out_dir: str | Path):
    """Line plot: number of earthquakes per year."""
    out_path = _setup_output_dir(out_dir) / "03_num_year.png"
    yearly_counts = df.groupby("year")["id"].count()

    plt.figure()
    plt.plot(yearly_counts.index, yearly_counts.values)
    plt.title("Number of Earthquakes by Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_depth_vs_mag(df: pd.DataFrame, out_dir: str | Path):
    """Scatter plot of depth vs magnitude."""
    out_path = _setup_output_dir(out_dir) / "04_depth_vs_mag.png"

    plt.figure()
    sns.scatterplot(data=df, x="mag", y="depth_km", alpha=0.5)
    plt.title("Depth vs. Magnitude")
    plt.xlabel("Magnitude")
    plt.ylabel("Depth (km)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, out_dir: str | Path):
    """Correlation heatmap for selected numerical columns."""
    out_path = _setup_output_dir(out_dir) / "05_corre.png"
    cols = [
        "nst",
        "gap",
        "dmin",
        "rms",
        "horizontalError",
        "depthError",
        "magError",
    ]
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()