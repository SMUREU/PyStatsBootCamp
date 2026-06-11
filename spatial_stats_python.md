# Introduction to Spatial Statistics

## NSF REU Bootcamp

By: Dr. Monnie McGee

Adapted to Python for the SMUREU Data Science Bootcamp.

---

# Before You Arrive: Install These Packages

This Python version uses:


```python
# If needed, install these in your environment before running the notebook:
# pip install geopandas libpysal esda splot scikit-gstat pykrige shapely scipy statsmodels geodatasets
```


```python
pip install geopandas libpysal esda splot scikit-gstat pykrige shapely scipy statsmodels geodatasets
```


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Point, LineString, box

import libpysal
from libpysal.weights import Queen, KNN, lag_spatial
from esda import Moran, Moran_Local

from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KernelDensity

# Optional geostatistics packages
try:
    import skgstat as skg
except ImportError:
    skg = None

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None

```

All mapping in this notebook uses **GeoPandas**.

---

# PART 1: What Makes Spatial Data Special?

## 1.1 Tobler's First Law of Geography

> *"Everything is related to everything else, but near things are more related than distant things."*
> — Waldo Tobler, 1970

This is the single most important idea in spatial statistics. Before writing any code, discuss:

- **Discussion prompt:** Think of two variables measured across a US state, say, median income and education level across counties. Would you expect nearby counties to look similar or different? Why?
- What assumption does this violate in classical statistics?

The violation is **independence**. Standard regression assumes observations are i.i.d. Spatial data are almost never independent they are *autocorrelated*.

## 1.2 Types of Spatial Data

There are three main flavors. Each requires different methods:

| Type | Description | Example |
|------|-------------|---------|
| **Areal / Lattice** | Observations tied to polygons | Counties, census tracts |
| **Point Process** | Locations of events | Disease cases, earthquakes |
| **Geostatistical** | Continuous field sampled at points | Soil zinc, rainfall, pollution |

Today we will touch all three.

---

# PART 2: Areal Data & Spatial Autocorrelation

## 2.1 Our Dataset: US County-Level SIDS Data

We will use the North Carolina SIDS dataset that is often used in spatial statistics examples. It contains county-level sudden infant death syndrome data and is a workhorse of spatial statistics pedagogy.


```python
# Load NC shapefile with SIDS data.
# This uses PySAL's example datasets.

example = libpysal.examples.load_example("sids2")
shp_path = example.get_path("sids2.shp")

nc = gpd.read_file(shp_path)

# Standardize column names if needed
nc.columns = [col.upper() if col.lower() in ["sid74", "bir74", "nwbir74", "sid79", "bir79", "nwbir79"] else col for col in nc.columns]

nc.head()
```


```python
nc.info()

```

**Key variables:**

- `SID74`: SIDS deaths 1974–78
- `BIR74`: live births 1974–78
- `NWBIR74`: non-white births 1974–78

Let's compute a SIDS rate per 1,000 births:


```python
nc["sids_rate74"] = (nc["SID74"] / nc["BIR74"]) * 1000
nc["sids_rate79"] = (nc["SID79"] / nc["BIR79"]) * 1000
nc["pct_nw74"] = (nc["NWBIR74"] / nc["BIR74"]) * 100

nc["sids_rate74"].describe()

```

## 2.2 Make a Choropleth Map


```python
ax = nc.plot(
    column="sids_rate74",
    legend=True,
    cmap="magma",
    edgecolor="white",
    linewidth=0.2,
    figsize=(9, 6),
    legend_kwds={"label": "SIDS rate\n(per 1,000 births)"}
)

ax.set_title("NC SIDS Rate, 1974–78")
ax.set_axis_off()
plt.show()

```

> **Discussion:** Do you see any spatial pattern? Are high-rate counties clustered, or scattered randomly?

## 2.3 Building a Spatial Weights Matrix

To measure spatial relationships formally, we need to define *who is a neighbor of whom*.

**Queen contiguity**: counties sharing any boundary point are neighbors (like a queen in chess).


```python
# Create queen-contiguity neighbors
w_queen = Queen.from_dataframe(nc)

# Row-standardize weights so each row sums to 1
w_queen.transform = "r"

print(w_queen)
print("Number of nonzero links:", w_queen.s0)
print("Average number of neighbors:", np.mean([len(v) for v in w_queen.neighbors.values()]))

```

Visualize the neighbor network.


```python
# Use centroids for drawing neighbor links
centroids = nc.geometry.centroid

lines = []

for i, neighbors in w_queen.neighbors.items():
    for j in neighbors:
        if i < j:
            lines.append(
                LineString([
                    centroids.iloc[i],
                    centroids.iloc[j]
                ])
            )

neighbor_lines = gpd.GeoDataFrame(
    geometry=lines,
    crs=nc.crs
)

ax = nc.plot(
    color="white",
    edgecolor="gray",
    linewidth=0.5,
    figsize=(9, 6)
)

neighbor_lines.plot(
    ax=ax,
    color="steelblue",
    linewidth=0.8
)

ax.set_title("Queen Contiguity Neighbor Network")
ax.set_axis_off()
plt.show()

```

> **Key concept:** The weights matrix **W** encodes which observations are "near" each other. Entry W_ij > 0 means county j is a neighbor of county i. Row-standardization means the spatial lag of x at location i is the *average* of x among i's neighbors.

## Overall structure for the data


```python
neighbor_counts = pd.Series(
    {i: len(neigh) for i, neigh in w_queen.neighbors.items()},
    name="neighbor_count"
)

neighbor_counts.describe()

```


```python
neighbor_counts.value_counts().sort_index()

```

- The weights matrix is very sparse — most county pairs are not neighbors, which is expected.
- On average, each county touches about 5 others.
- The distribution table shows that some counties have only a few neighbors while others have many more.
- This makes geographic sense. Small or oddly shaped counties on the coast or state border tend to have fewer neighbors, while large central counties touch more.
- If we had any counties with 0 or 1 links that would be worth investigating.

## 2.4 The Spatial Lag

The **spatial lag** of a variable is its weighted average among neighbors. If a variable is spatially autocorrelated, its value and its spatial lag should be correlated.


```python
nc["sids_lag"] = lag_spatial(
    w_queen,
    nc["sids_rate74"]
)

```


```python
plt.figure(figsize=(7, 5))
plt.scatter(
    nc["sids_rate74"],
    nc["sids_lag"],
    alpha=0.7
)

# Add regression line
coef = np.polyfit(nc["sids_rate74"], nc["sids_lag"], 1)
x_vals = np.linspace(nc["sids_rate74"].min(), nc["sids_rate74"].max(), 100)
plt.plot(x_vals, coef[0] * x_vals + coef[1], color="tomato")

plt.xlabel("SIDS rate (county)")
plt.ylabel("Spatial lag (avg. of neighbors)")
plt.title("Moran Scatterplot: SIDS Rate 1974–78")
plt.show()

```

This is the **Moran scatterplot**. The slope of the line is Moran's I!

## 2.5 Moran's I: Formal Test for Spatial Autocorrelation

$$I = \frac{n}{\sum_{i}\sum_{j} w_{ij}} \cdot \frac{\sum_{i}\sum_{j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

- $I \approx 0$: no spatial autocorrelation (random)
- $I > 0$: positive autocorrelation (clusters of similar values)
- $I < 0$: negative autocorrelation (checkerboard pattern)


```python
moran_test = Moran(
    nc["sids_rate74"],
    w_queen,
    permutations=999
)

print("Moran's I:", moran_test.I)
print("Expected I:", moran_test.EI)
print("Permutation p-value:", moran_test.p_sim)

```

> **Interpret together:** Is I significantly positive? What does that mean substantively for SIDS rates across NC?

### Permutation-based inference

The analytical p-value assumes normality. A safer approach: permute the spatial arrangement 999 times.


```python
plt.figure(figsize=(7, 5))
plt.hist(
    moran_test.sim,
    bins=30,
    edgecolor="white"
)
plt.axvline(
    moran_test.I,
    color="red",
    linewidth=2,
    label="Observed Moran's I"
)
plt.xlabel("Moran's I under permutation")
plt.ylabel("Count")
plt.title("Permutation Distribution of Moran's I")
plt.legend()
plt.show()

```

## 2.6 Local Moran's I (LISA)

Global Moran's I gives one number for the whole map. **Local** indicators (LISA) tell us *where* clusters and outliers are.

Four quadrant types in the Moran scatterplot:

- **High-High (HH)**: hot spots — high surrounded by high
- **Low-Low (LL)**: cold spots — low surrounded by low
- **High-Low (HL)**: outlier — high surrounded by low
- **Low-High (LH)**: outlier — low surrounded by high


```python
lisa = Moran_Local(
    nc["sids_rate74"],
    w_queen,
    permutations=999
)

nc["local_I"] = lisa.Is
nc["local_p"] = lisa.p_sim

```

Classify into LISA categories.


```python
nc_mean = nc["sids_rate74"].mean()

conditions = [
    (nc["local_p"] < 0.05) & (nc["sids_rate74"] > nc_mean) & (nc["sids_lag"] > nc_mean),
    (nc["local_p"] < 0.05) & (nc["sids_rate74"] < nc_mean) & (nc["sids_lag"] < nc_mean),
    (nc["local_p"] < 0.05) & (nc["sids_rate74"] > nc_mean) & (nc["sids_lag"] < nc_mean),
    (nc["local_p"] < 0.05) & (nc["sids_rate74"] < nc_mean) & (nc["sids_lag"] > nc_mean)
]

choices = [
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High"
]

nc["lisa_cat"] = np.select(
    conditions,
    choices,
    default="Not significant"
)

nc["lisa_cat"].value_counts()

```


```python
lisa_colors = {
    "High-High": "#d73027",
    "Low-Low": "#4575b4",
    "High-Low": "#fc8d59",
    "Low-High": "#91bfdb",
    "Not significant": "lightgray"
}

ax = nc.plot(
    color=nc["lisa_cat"].map(lisa_colors),
    edgecolor="white",
    linewidth=0.2,
    figsize=(9, 6)
)

# Add a custom legend
handles = [
    plt.Line2D([0], [0], marker="s", color="w", label=label,
               markerfacecolor=color, markersize=10)
    for label, color in lisa_colors.items()
]

ax.legend(
    handles=handles,
    title="LISA cluster",
    loc="lower left"
)

ax.set_title("Local Moran's I — SIDS Rate Clusters\np < 0.05; NC counties 1974–78")
ax.set_axis_off()
plt.show()

```

> **Discussion:** Where are the hot spots? Do they correspond to any geographic features or demographic patterns you might expect?

---

# PART 3: Point Patterns

## 3.1 Are Events Clustered, Random, or Dispersed?

Point process analysis asks: given a set of event locations, is their spatial distribution consistent with **complete spatial randomness (CSR)**, or is there evidence of clustering/repulsion?

We'll use **earthquake epicenters** from the `quakes` dataset in base R — 1000 earthquakes near Fiji.


```python
# Load quakes data.
# First try a local file. If unavailable, use statsmodels' R dataset loader.
try:
    quakes = pd.read_csv("../data/quakes.csv")
except FileNotFoundError:
    import statsmodels.api as sm
    quakes = sm.datasets.get_rdataset("quakes", "datasets").data

quakes.head()

```


```python
quakes_gdf = gpd.GeoDataFrame(
    quakes,
    geometry=gpd.points_from_xy(
        quakes["long"],
        quakes["lat"]
    ),
    crs="EPSG:4326"
)

quakes_gdf.head()

```


```python
ax = quakes_gdf.plot(
    column="depth",
    markersize=quakes_gdf["mag"] ** 2,
    alpha=0.4,
    legend=True,
    cmap="plasma",
    figsize=(8, 6),
    legend_kwds={"label": "Depth (km)"}
)

ax.set_title("Earthquakes Near Fiji (M ≥ 4)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

```

## 3.2 Kernel Density Estimation

Rather than plotting raw points, we can estimate the *intensity* of events across space.


```python
# Estimate a spatial KDE over longitude and latitude.

coords = quakes_gdf[["long", "lat"]].to_numpy()

kde = KernelDensity(
    bandwidth=0.25,
    kernel="gaussian"
)

kde.fit(coords)

xmin, ymin, xmax, ymax = quakes_gdf.total_bounds

x_grid = np.linspace(xmin, xmax, 150)
y_grid = np.linspace(ymin, ymax, 150)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_coords = np.column_stack([
    xx.ravel(),
    yy.ravel()
])

density = np.exp(kde.score_samples(grid_coords))
density_grid = density.reshape(xx.shape)
```

Convert the KDE grid into a GeoPandas object for mapping.


```python
cell_width = x_grid[1] - x_grid[0]
cell_height = y_grid[1] - y_grid[0]

grid_polygons = []
grid_values = []

for i in range(len(x_grid) - 1):
    for j in range(len(y_grid) - 1):
        grid_polygons.append(
            box(
                x_grid[i],
                y_grid[j],
                x_grid[i + 1],
                y_grid[j + 1]
            )
        )
        grid_values.append(density_grid[j, i])

kde_gdf = gpd.GeoDataFrame(
    {"density": grid_values},
    geometry=grid_polygons,
    crs=quakes_gdf.crs
)

```


```python
ax = kde_gdf.plot(
    column="density",
    cmap="inferno",
    legend=True,
    figsize=(8, 6),
    edgecolor=None,
    alpha=0.9,
    legend_kwds={"label": "Density"}
)

quakes_gdf.plot(
    ax=ax,
    color="white",
    markersize=0.5,
    alpha=0.3
)

ax.set_title("Kernel Density of Earthquake Epicenters")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

```

> **Discussion:** What patterns do you see? Why might earthquakes cluster near Fiji?

## 3.3 Quadrat Test for CSR (Brief)

The simplest test: divide the study region into cells, count events per cell, test against Poisson.


```python
# Conceptual chi-squared test comparing observed cell counts to expected under CSR.

n_x = 5
n_y = 5

quakes_gdf["x_bin"] = pd.cut(
    quakes_gdf["long"],
    bins=np.linspace(xmin, xmax, n_x + 1),
    labels=False,
    include_lowest=True
)

quakes_gdf["y_bin"] = pd.cut(
    quakes_gdf["lat"],
    bins=np.linspace(ymin, ymax, n_y + 1),
    labels=False,
    include_lowest=True
)

observed_counts = (
    quakes_gdf
    .groupby(["x_bin", "y_bin"])
    .size()
    .reindex(
        pd.MultiIndex.from_product(
            [range(n_x), range(n_y)],
            names=["x_bin", "y_bin"]
        ),
        fill_value=0
    )
)

expected_count = len(quakes_gdf) / (n_x * n_y)

chi2_stat = ((observed_counts - expected_count) ** 2 / expected_count).sum()
df = n_x * n_y - 1
p_value = 1 - stats.chi2.cdf(chi2_stat, df)

print("Quadrat chi-squared statistic:", chi2_stat)
print("Degrees of freedom:", df)
print("p-value:", p_value)

```


```python
# Map quadrat counts using GeoPandas

quadrat_polygons = []
quadrat_counts = []

x_edges = np.linspace(xmin, xmax, n_x + 1)
y_edges = np.linspace(ymin, ymax, n_y + 1)

for i in range(n_x):
    for j in range(n_y):
        quadrat_polygons.append(
            box(
                x_edges[i],
                y_edges[j],
                x_edges[i + 1],
                y_edges[j + 1]
            )
        )
        quadrat_counts.append(observed_counts.loc[(i, j)])

quadrat_gdf = gpd.GeoDataFrame(
    {"count": quadrat_counts},
    geometry=quadrat_polygons,
    crs=quakes_gdf.crs
)

ax = quadrat_gdf.plot(
    column="count",
    legend=True,
    cmap="viridis",
    edgecolor="white",
    linewidth=0.5,
    figsize=(8, 6),
    legend_kwds={"label": "Earthquake count"}
)

quakes_gdf.plot(
    ax=ax,
    color="black",
    markersize=1,
    alpha=0.3
)

ax.set_title("Quadrat Counts for Earthquake Epicenters")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

```


```python
print("For a rigorous point process analysis, explore point-pattern packages such as pointpats.")
print("Related concepts: K function, L function, quadrat tests, spatial envelopes.")

```

**Exercise (take-home):** Explore a Python point-pattern package such as `pointpats`, convert the quakes data to a point pattern, and compute Ripley's K or L function. Does it suggest clustering at short distances?

---

# PART 4: Geostatistics & Kriging

## 4.1 The Variogram: How Does Similarity Decay with Distance?

For geostatistical data (continuous surface, sampled at points), we model how the **variance** between pairs of measurements grows with distance.

$$\gamma(h) = \frac{1}{2|N(h)|} \sum_{N(h)} [Z(s_i) - Z(s_j)]^2$$

where $N(h)$ is the set of pairs separated by distance $h$.

Three key variogram parameters:

- **Nugget**: variance at distance 0 (measurement error + microscale variation)
- **Sill**: total variance (where the variogram levels off)
- **Range**: distance at which spatial correlation effectively vanishes

We'll use the `meuse` river dataset: heavy metal concentrations in floodplain soils.


```python
# Load meuse and meuse.grid.
# First try local files. If unavailable, load R datasets via statsmodels.

try:
    meuse = pd.read_csv("meuse.csv")
    meuse_grid = pd.read_csv("meuse_grid.csv")
except FileNotFoundError:
    import statsmodels.api as sm
    meuse = sm.datasets.get_rdataset("meuse", "sp").data
    meuse_grid = sm.datasets.get_rdataset("meuse.grid", "sp").data

meuse.head()

```


```python
meuse_sf = gpd.GeoDataFrame(
    meuse,
    geometry=gpd.points_from_xy(
        meuse["x"],
        meuse["y"]
    ),
    crs="EPSG:28992"
)

meuse_grid_sf = gpd.GeoDataFrame(
    meuse_grid,
    geometry=gpd.points_from_xy(
        meuse_grid["x"],
        meuse_grid["y"]
    ),
    crs="EPSG:28992"
)

meuse_sf["log_zinc"] = np.log(meuse_sf["zinc"])

meuse_sf.head()

```


```python
ax = meuse_sf.plot(
    column="log_zinc",
    markersize=meuse_sf["log_zinc"] * 15,
    alpha=0.8,
    legend=True,
    cmap="turbo",
    figsize=(8, 6),
    legend_kwds={"label": "log(Zinc)"}
)

ax.set_title("Zinc Concentration in Meuse Floodplain\nSampled soil locations; log scale")
ax.set_axis_off()
plt.show()
```

## 4.2 Compute the Empirical Variogram


```python
coords = np.column_stack([
    meuse_sf.geometry.x,
    meuse_sf.geometry.y
])

values = meuse_sf["log_zinc"].to_numpy()

```


```python
# Empirical variogram using scikit-gstat if available.

if skg is not None:
    v_emp = skg.Variogram(
        coords,
        values,
        normalize=False,
        n_lags=15,
        model="spherical"
    )

    fig = v_emp.plot()
    plt.title("Empirical Variogram: log(Zinc)")
    plt.show()

else:
    # Manual fallback empirical variogram
    distances = pdist(coords)
    semivariance = 0.5 * pdist(values.reshape(-1, 1), metric="sqeuclidean")

    bins = np.linspace(0, distances.max(), 16)
    bin_ids = np.digitize(distances, bins)

    rows = []
    for b in range(1, len(bins)):
        mask = bin_ids == b
        if mask.sum() > 0:
            rows.append({
                "distance": distances[mask].mean(),
                "semivariance": semivariance[mask].mean(),
                "count": mask.sum()
            })

    v_emp = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.scatter(v_emp["distance"], v_emp["semivariance"])
    plt.xlabel("Distance")
    plt.ylabel("Semivariance")
    plt.title("Empirical Variogram: log(Zinc)")
    plt.show()

```

> **Discussion:** Does the variogram level off? At roughly what distance (in meters)? What does the nugget look like?

## 4.3 Fit a Variogram Model


```python
# Fit a spherical model.

if skg is not None:
    v_fit = skg.Variogram(
        coords,
        values,
        normalize=False,
        n_lags=15,
        model="spherical",
        use_nugget=True
    )

    print(v_fit)
    print("Parameters:", v_fit.parameters)

    fig = v_fit.plot()
    plt.title("Fitted Spherical Variogram: log(Zinc)")
    plt.show()

else:
    print("scikit-gstat is not installed. Install scikit-gstat to fit a variogram model directly.")

```


```python
print("The total variance is", round(np.var(meuse_sf["log_zinc"], ddof=1), 3))

```

### Explaining the output

#### Row 1 — Nugget (Nug)

In the R version, the fitted nugget is around 0.05: the variance at zero distance, which is tiny relative to the total. This represents measurement error and microscale variation in zinc that occurs over distances shorter than the closest sample spacing. A small nugget like this is good news: it means most of the variance in log(zinc) is spatially structured, not noise.

#### Row 2 — Spherical component (Sph)

In the R version:

- psill ≈ 0.59: the spatially structured variance
- range ≈ 897 meters: spatial correlation in log(zinc) effectively disappears beyond about 900 meters

Two soil samples more than 900m apart are essentially uncorrelated.

The total sill is nugget + psill ≈ 0.05 + 0.59 = 0.64.

#### The big picture

The ratio of structured variance to total variance is 0.59 / 0.64 ≈ 92%. This means nearly all the variability in log(zinc) across the floodplain is spatially structured. Zinc concentrations change smoothly and predictably with location, rather than jumping around randomly. That's exactly what makes kriging worthwhile here: there is strong spatial signal to exploit when making predictions. Notice also that the starting values can be nudged slightly by least-squares fitting. That's the fitting procedure doing its job.

A sill somewhat higher than the sample variance is actually common and not necessarily a problem. The empirical variogram is computed from pairwise differences across the whole dataset, while `var()` is computed around the global mean. If the data have a spatial trend (and zinc almost certainly does), those two quantities measure slightly different things. The variogram can exceed the sample variance when a trend is present.


```python
ax = meuse_sf.plot(
    column="log_zinc",
    markersize=meuse_sf["log_zinc"] * 15,
    legend=True,
    cmap="turbo",
    figsize=(8, 6),
    legend_kwds={"label": "log(Zinc)"}
)

ax.set_title("Checking for Spatial Trend in log(Zinc)")
ax.set_axis_off()
plt.show()

```

A clear gradient (higher values near the river channel) confirms a trend. In that case the right move is to fit the variogram on the residuals after removing the trend. In the original R version, this is done using distance to the river as a covariate.


```python
# Use distance to river as a covariate if the column exists.

if "dist" in meuse_sf.columns:
    import statsmodels.formula.api as smf

    trend_model = smf.ols(
        "log_zinc ~ dist",
        data=meuse_sf
    ).fit()

    meuse_sf["log_zinc_resid"] = trend_model.resid

    resid_values = meuse_sf["log_zinc_resid"].to_numpy()

    if skg is not None:
        v_emp_dist = skg.Variogram(
            coords,
            resid_values,
            normalize=False,
            n_lags=15,
            model="spherical",
            use_nugget=True
        )

        print(v_emp_dist)
        print("Parameters:", v_emp_dist.parameters)

        fig = v_emp_dist.plot()
        plt.title("Fitted Spherical Variogram: residual log(Zinc) after distance trend")
        plt.show()

else:
    print("No 'dist' column found in the Meuse data.")

```

## 4.4 Ordinary Kriging Prediction

Now we interpolate to a full grid — this is **ordinary kriging**.


```python
# Prepare prediction grid

gridx = np.sort(meuse_grid_sf.geometry.x.unique())
gridy = np.sort(meuse_grid_sf.geometry.y.unique())

```


```python
if OrdinaryKriging is not None:
    OK = OrdinaryKriging(
        meuse_sf.geometry.x.to_numpy(),
        meuse_sf.geometry.y.to_numpy(),
        meuse_sf["log_zinc"].to_numpy(),
        variogram_model="spherical"
    )

    z_pred, z_var = OK.execute(
        "grid",
        gridx,
        gridy
    )

    xx, yy = np.meshgrid(gridx, gridy)

    krig_sf = gpd.GeoDataFrame(
        {
            "var1.pred": z_pred.ravel(),
            "var1.var": z_var.ravel()
        },
        geometry=gpd.points_from_xy(
            xx.ravel(),
            yy.ravel()
        ),
        crs=meuse_sf.crs
    )

else:
    # Fallback interpolation if pykrige is unavailable.
    xx, yy = np.meshgrid(gridx, gridy)

    z_pred = griddata(
        coords,
        meuse_sf["log_zinc"].to_numpy(),
        (xx, yy),
        method="cubic"
    )

    z_var = np.full_like(z_pred, np.nan)

    krig_sf = gpd.GeoDataFrame(
        {
            "var1.pred": z_pred.ravel(),
            "var1.var": z_var.ravel()
        },
        geometry=gpd.points_from_xy(
            xx.ravel(),
            yy.ravel()
        ),
        crs=meuse_sf.crs
    )

krig_sf.head()

```


```python
ax = krig_sf.dropna(subset=["var1.pred"]).plot(
    column="var1.pred",
    legend=True,
    cmap="turbo",
    markersize=8,
    figsize=(8, 6),
    legend_kwds={"label": "Kriged\nlog(Zinc)"}
)

meuse_sf.plot(
    ax=ax,
    color="white",
    markersize=25,
    marker="+"
)

ax.set_title("Ordinary Kriging Prediction: log(Zinc)\nCrosses = sample locations")
ax.set_axis_off()
plt.show()

```


```python
# Kriging also gives prediction variance (uncertainty!)

if krig_sf["var1.var"].notna().any():
    ax = krig_sf.plot(
        column="var1.var",
        legend=True,
        cmap="magma",
        markersize=8,
        figsize=(8, 6),
        legend_kwds={"label": "Kriging\nvariance"}
    )

    ax.set_title("Kriging Prediction Variance\nHigher variance = less certain (far from sample points)")
    ax.set_axis_off()
    plt.show()

else:
    print("Prediction variance is only available when pykrige is installed.")

```

### Kriging with distance as covariate

The R version performs universal kriging with distance-to-river as a trend.

In Python, a simple equivalent workflow is:

1. Fit the trend model `log_zinc ~ dist`
2. Krige the residuals
3. Add the predicted trend back to the kriged residual surface


```python
if "dist" in meuse_sf.columns and OrdinaryKriging is not None:
    trend_model = smf.ols(
        "log_zinc ~ dist",
        data=meuse_sf
    ).fit()

    meuse_sf["trend_resid"] = trend_model.resid

    OK_resid = OrdinaryKriging(
        meuse_sf.geometry.x.to_numpy(),
        meuse_sf.geometry.y.to_numpy(),
        meuse_sf["trend_resid"].to_numpy(),
        variogram_model="spherical"
    )

    # Use the actual prediction points from meuse_grid_sf
    grid_x = meuse_grid_sf.geometry.x.to_numpy()
    grid_y = meuse_grid_sf.geometry.y.to_numpy()

    resid_pred, resid_var = OK_resid.execute(
        "points",
        grid_x,
        grid_y
    )

    # Predict trend at the same prediction points
    if "dist" in meuse_grid_sf.columns:
        grid_dist = meuse_grid_sf["dist"].to_numpy()
    else:
        grid_dist = np.zeros(len(meuse_grid_sf))

    trend_pred = trend_model.predict(
        pd.DataFrame({"dist": grid_dist})
    ).to_numpy()

    universal_pred = trend_pred + np.asarray(resid_pred)

    krig_dist_sf = gpd.GeoDataFrame(
        {
            "var1.pred": universal_pred,
            "var1.var": np.asarray(resid_var)
        },
        geometry=meuse_grid_sf.geometry,
        crs=meuse_sf.crs
    )

else:
    krig_dist_sf = None
    print("Universal kriging-style example requires a 'dist' column and pykrige.")
```


```python
if krig_dist_sf is not None:
    ax = krig_dist_sf.plot(
        column="var1.pred",
        legend=True,
        cmap="turbo",
        markersize=8,
        figsize=(8, 6),
        legend_kwds={"label": "Kriged\nlog(Zinc)"}
    )

    meuse_sf.plot(
        ax=ax,
        color="white",
        markersize=25,
        marker="+"
    )

    ax.set_title("Universal Kriging Prediction: log(Zinc)\nTrend: distance to river | Crosses = sample locations")
    ax.set_axis_off()
    plt.show()

```


```python
if krig_dist_sf is not None:
    ax = krig_dist_sf.plot(
        column="var1.var",
        legend=True,
        cmap="magma",
        markersize=8,
        figsize=(8, 6),
        legend_kwds={"label": "Kriging\nvariance"}
    )

    ax.set_title("Universal Kriging Prediction Variance\nHigher variance = less certain (far from sample points)")
    ax.set_axis_off()
    plt.show()

```

> **Key insight:** Kriging is a **BLUP** (Best Linear Unbiased Predictor). It provides not just predictions but also *uncertainty estimates*. Notice that variance is lowest near sample points and highest in data-sparse regions.

> **Always map the raw data:** Before fitting any variogram, if you see a directional gradient like for the Meuse Zinc concentration data, ordinary kriging is the wrong choice. The map tells you the model.

---

# PART 5: Bringing It Together — Discussion & Next Steps

## What We Covered

| Concept | Method | Python Tools |
|---------|--------|--------------|
| Spatial autocorrelation | Moran's I, LISA | `esda.Moran`, `esda.Moran_Local` |
| Neighbor structure | Weights matrix | `libpysal.weights.Queen`, `KNN` |
| Point patterns | KDE, CSR tests | `KernelDensity`, quadrat counts, `pointpats` |
| Spatial interpolation | Kriging | `scikit-gstat`, `pykrige` |
| Mapping | Choropleths, point maps, prediction maps | `geopandas` |

## What Comes Next

1. **Spatial regression** — when OLS residuals are spatially autocorrelated, use spatial lag or spatial error models (`spreg`)
2. **Spatial scan statistics** — cluster detection for public health (`SaTScan`, `rsatscan`, or Python alternatives)
3. **Bayesian spatial models** — spatial hierarchical models
4. **Machine learning + space** — random forests with spatial cross-validation

## Discussion Questions

1. We found spatial clustering in SIDS rates. Does that *prove* a spatial process caused it? What alternative explanations exist?

2. We used queen contiguity for the weights matrix. How might results change with k-nearest-neighbor weights? Distance-band weights?

3. Kriging assumes stationarity — the variogram is the same everywhere. When might this fail?

---

# EXERCISES (To complete during or after the session)

## Exercise 1: Repeat the Moran's I analysis for SIDS rates in 1979 (`sids_rate79`). Compare the Moran's I values between 1974–78 and 1979–84. What do you find?


```python
# Your code here

# moran_1979 = Moran(
#     nc["sids_rate79"],
#     w_queen,
#     permutations=999
# )
#
# print(moran_1979.I)
# print(moran_1979.p_sim)
```

## Exercise 2: Change the neighbor definition to k=5 nearest neighbors. Does Moran's I change substantially?


```python
# Coordinates from county centroids
coords = np.column_stack([
    nc.geometry.centroid.x,
    nc.geometry.centroid.y
])

w_knn5 = KNN.from_array(
    coords,
    k=5
)

w_knn5.transform = "r"

moran_knn5 = Moran(
    nc["sids_rate74"],
    w_knn5,
    permutations=999
)

print(moran_knn5.I)
print(moran_knn5.p_sim)
```

## Exercise 3: For the meuse data, fit an **exponential** variogram model instead of spherical. Which fits better? Compare model fit diagnostics.


```python
# Your code here

# if skg is not None:
#     v_exp = skg.Variogram(
#         coords,
#         values,
#         normalize=False,
#         n_lags=15,
#         model="exponential",
#         use_nugget=True
#     )
#
#     print(v_exp)
#     print(v_exp.parameters)
#
#     fig = v_exp.plot()
#     plt.title("Fitted Exponential Variogram: log(Zinc)")
#     plt.show()

```

## Exercise 4 (Challenge): Use a Python point-pattern package and compute the **L-function** or a related clustering statistic. Is there evidence of clustering at short distances?


```python
# Possible package to explore:
# pip install pointpats

# Example direction:
# from pointpats import PointPattern
# pp = PointPattern(quakes_gdf[["long", "lat"]].to_numpy())
# Then explore K or L function tools available in the package.
```
