"""
clustering.py
~~~~~~~~~~~~~
Functions for spatial clustering of earthquake epicenters
and visualisation on a Folium map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import folium
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: Tuple[str, str] = ("latitude", "longitude"),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply K‑Means clustering and return the dataframe with a 'cluster' column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned earthquake dataframe.
    n_clusters : int
        Number of clusters.
    features : Tuple[str, str]
        Column names used for clustering.
    """
    coords = df[list(features)].values
    labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(coords)
    df = df.copy()
    df["cluster"] = labels
    return df


def plot_kmeans_2d(
    df: pd.DataFrame,
    out_dir: str | Path,
    features: Tuple[str, str] = ("longitude", "latitude"),
):
    """2‑D scatter plot coloured by cluster."""
    out_path = Path(out_dir) / "06_kmeans_2d.png"
    plt.figure()
    sns.scatterplot(
        data=df,
        x=features[0],
        y=features[1],
        hue="cluster",
        palette="tab10",
        s=20,
        alpha=0.6,
    )
    plt.title("K‑Means Clustering (2‑D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_folium_map(df: pd.DataFrame, out_html: str | Path):
    """
    Create an interactive Folium map highlighting high‑magnitude earthquakes.

    Only events with mag >= 6 are plotted to keep the map lightweight.
    """
    m = folium.Map(location=[38, 142], zoom_start=4, tiles="CartoDB positron")

    cluster_colors = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#6a3d9a",
        "#b15928",
    ]

    for _, row in df[df["mag"] >= 6].iterrows():
        tooltip = (
            f"<b>Mag:</b> {row['mag']}<br/>"
            f"<b>Depth:</b> {row['depth_km']} km<br/>"
            f"<b>Cluster:</b> {row['cluster']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color="white",
            weight=0.4,
            fill=True,
            fill_color=cluster_colors[int(row["cluster"]) % len(cluster_colors)],
            fill_opacity=0.6,
            tooltip=tooltip,
        ).add_to(m)

    m.save(out_html)
    print(f"Folium map saved to: {out_html}")