# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **EDA**

# %% [markdown]
# # **Checking Information of Dataset**

# %%
import pandas as pd
df = pd.read_csv('source.csv')

print(df.head())
print(df.info())
print(df.describe())

# %% [markdown]
# # **Convert to Datetime**

# %%
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['updated'] = pd.to_datetime(df['updated'], errors='coerce')

# %% [markdown]
# # **Handling Missing Earthquake Data in DataFrame**

# %%
# Insert checks and handling of missing values here
print("Total missing values:\n", df.isnull().sum())

original_len = len(df)

# drop rows if latitude, longitude, mag, or depth are missing
df.dropna(subset=['latitude', 'longitude', 'mag', 'depth'], inplace=True)

# Check how many rows remain after dropping
print(f"DataFrame length before dropna: {original_len}")
print(f"DataFrame length after dropna: {len(df)}")

# %% [markdown]
# There was no missing value.

# %% [markdown]
# # **Visualizing Earthquake Magnitudes, Depths, Temporal Trends, and Epicenters**

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=df, x='mag', binwidth=0.2)
plt.title('Distribution of Earthquake Magnitudes')
plt.show()

# %% [markdown]
# Most earthquakes have magnitudes between 4.0 and 5.0. As magnitude increases, the number of events drops sharply, showing a right-skewed distribution.

# %%
sns.histplot(data=df, x='depth', binwidth=10)
plt.title('Distribution of Earthquake Depths')
plt.show()

# %% [markdown]
# Most earthquakes occur at shallow depths, mainly between 0 and 100 km. Deeper earthquakes are much less frequent, showing a right-skewed distribution

# %%
df['year'] = df['time'].dt.year
yearly_counts = df.groupby('year')['id'].count()
plt.plot(yearly_counts.index, yearly_counts.values)
plt.title('Number of Earthquakes by Year')
plt.show()

# %% [markdown]
# # **Scatterplot of Depth vs. Magnitude**

# %%
sns.scatterplot(data=df, x='mag', y='depth', alpha=0.5)
plt.title('Depth vs. Magnitude')
plt.show()

# %% [markdown]
# Most earthquakes occur at shallow depths. There is no clear correlation between depth and magnitude.

# %%
corr = df[['nst','gap','dmin','rms','horizontalError','depthError','magError']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# * There is a strong negative correlation between nst (number of stations) and gap (-0.66):
#   * → When more stations are available, the azimuthal gap becomes smaller, indicating better coverage and likely higher data accuracy.
# * nst also shows a moderate negative correlation with depthError (-0.37):
#   * → More stations contribute to more accurate depth estimation. This suggests improving station density could reduce depth estimation error.
# * gap is moderately positively correlated with magError (0.44):
#   * → A larger azimuthal gap may lead to less accurate magnitude calculations, likely due to incomplete directional coverage.
# * dmin (minimum distance to station) is positively correlated with horizontalError (0.52):
#   * → The farther the station is from the epicenter, the more likely horizontal location error increases. This emphasizes the importance of having nearby sensors.
# * magError is weakly positively correlated with several variables, such as nst (0.44), depthError (0.22), and horizontalError (0.083):
#   * → While not extremely strong, it indicates that multiple observational factors can collectively influence magnitude accuracy.

# %% [markdown]
# # **Machine Learning**

# %% [markdown]
# # **Clustering of Earthquakes (Spatial Pattern Analysis)**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

df = pd.read_csv("source.csv")

# Clustering Columns
data_for_cluster = df[['latitude', 'longitude', 'depth']].dropna()

# Standarize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_cluster)

# KMeans
k = 4 # the number of plates
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(data_scaled)


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude', 'depth'])
centroids['cluster_id'] = range(k)
print("Cluster Centers (standardized):\n", centroids)

# Add labels
clustered_df = data_for_cluster.copy()
clustered_df['cluster_label'] = labels

print("\nCluster Means:")
print(clustered_df.groupby('cluster_label').mean())


cluster_color_map = {
    0: '#1f78b4',
    1: '#33a02c',
    2: '#e31a1c',
    3: '#ff7f00' 
}

# クラスタラベルに基づいて色を割り当て
color_list = [cluster_color_map[label] for label in labels]

# 2D 可視化（緯度・経度）
plt.figure(figsize=(8, 6))
plt.scatter(
    data_for_cluster['longitude'],
    data_for_cluster['latitude'],
    c=color_list,
    alpha=0.6,
    edgecolors='white',
    linewidths=0.5
)
plt.title(f"KMeans Clustering (k={k})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 3D 可視化（緯度・経度・深さ）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    data_for_cluster['longitude'],
    data_for_cluster['latitude'],
    data_for_cluster['depth'],
    c=color_list,
    edgecolors='white',
    linewidths=0.5
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth")
plt.title(f"KMeans Clustering (k={k})")
plt.show()

# %% [markdown]
# # **Mapping Major Earthquakes in Japan with Folium**

# %%
import folium

# Folium map
m = folium.Map(location=[38, 142], zoom_start=4, tiles='Esri.WorldImagery')

cluster_colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']

for _, row in clustered_df.iterrows():
    tooltip = f"""
    <b>Depth:</b> {row['depth']} km<br/>
    <b>Cluster:</b> {row['cluster_label']}
    """
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='white',
        weight=0.4,
        fill=True,
        fill_color=cluster_colors[int(row['cluster_label'])],
        fill_opacity=0.4,
        tooltip=tooltip
    ).add_to(m)

# Legend
legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; width: 180px; height: 140px;
    background-color: white; z-index:9999; font-size:14px;
    border:2px solid grey; padding:10px;">
    <b>Cluster Label</b><br>
    <i style="background:#e31a1c; width:10px; height:10px; display:inline-block;"></i> Cluster 0<br>
    <i style="background:#1f78b4; width:10px; height:10px; display:inline-block;"></i> Cluster 1<br>
    <i style="background:#ff7f00; width:10px; height:10px; display:inline-block;"></i> Cluster 2<br>
    <i style="background:#33a02c; width:10px; height:10px; display:inline-block;"></i> Cluster 3
    
    
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))
m
