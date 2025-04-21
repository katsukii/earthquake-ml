[![pages](https://img.shields.io/badge/GitHub-Pages-blue)](https://katsukii.github.io/earthquake-ml-project/)

# Seismic Insight Machine Learning

## Overview

This project explores **21,000+ historical earthquakes in and around Japan** to uncover
spatial and temporal patterns relevant to disaster‑prevention research.

- Cleaned and enriched raw USGS data (datetime parsing and missing‑value handling).
- Performed exploratory data analysis (magnitude, depth, yearly trend).
- Applied **K‑Means clustering** (k = 4) to reveal seismic zones that align with Japan’s four major tectonic plates.
- Built an **interactive Folium map** of high‑magnitude events (M ≥ 6).

## Results

| Figure                                   | Insight                                 |
| ---------------------------------------- | --------------------------------------- |
| ![](reports/figures/01_mag_hist.png)     | Magnitudes are right‑skewed.            |
| ![](reports/figures/02_depth_hist.png)   | Most quakes are shallow ( &lt; 100 km). |
| ![](reports/figures/03_num_year.png)     | Year‑by‑year trend in event counts.     |
| ![](reports/figures/04_depth_vs_mag.png) | No clear depth–magnitude correlation.   |
| ![](reports/figures/05_corre.png)        | Feature‑correlation heatmap.            |
| ![](reports/figures/06_kmeans_2d.png)    | K‑Means clusters (2‑D).                 |
| ![](reports/figures/07_kmeans_3d.png)    | K‑Means clusters (3‑D).                 |

For a full interactive report, visit the **[GitHub Pages site](https://katsukii.github.io/earthquake-ml-project/)**.

## Dataset

[Earthquakes in Japan – Kaggle](https://www.kaggle.com/datasets/aerodinamicc/earthquakes-in-japan)  

## Tech Stack

Python · pandas · scikit‑learn · seaborn · matplotlib · Folium
