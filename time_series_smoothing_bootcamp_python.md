---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Time Series Bootcamp

## Exponential Smoothing, Forecasting Workflow, and a Few Useful Tools

By: Dr. Monnie McGee

Adapted to Python for the SMUREU Data Science Bootcamp.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.nonparametric.smoothers_lowess import lowess

from scipy import stats
from scipy.stats import gaussian_kde
```

## Setup

The Python version of this lesson uses:

- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `scipy`

**Primary reference:** Hyndman, R.J. & Athanasopoulos, G. (2021),
*Forecasting: Principles and Practice* (3rd ed.).
<https://otexts.com/fpp3/>

---

## Time Series

```python
np.random.seed(123)

n = 200
y = np.zeros(n)

for t in range(1, n):
    y[t] = 0.8 * y[t - 1] + np.random.normal()

time = np.arange(1, n + 1)

plt.figure(figsize=(8, 4))
plt.plot(time, y)
plt.xlabel("Time")
plt.ylabel("y")
plt.title("Simulated autocorrelated time series")
plt.show()

fit_lm = sm.OLS(
    y,
    sm.add_constant(time)
).fit()

fit_lm.summary()
```

The true process has no linear trend. It has long runs above and below zero.

```python
plt.figure(figsize=(8, 4))
plt.plot(time, fit_lm.resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.title("Residuals from linear model")
plt.show()
```

- Do the residuals look independent?
- Are there clusters of positive and negative residuals?

We pull two publicly available, continuously updated series directly from their
authoritative sources:

* **Temperature anomalies** — NASA GISS Surface Temperature Analysis (GISTEMP v4),
  global land-ocean index, annual means, baseline 1951–1980. Updated monthly.
  <https://data.giss.nasa.gov/gistemp/>

* **Atmospheric CO₂** — NOAA Global Monitoring Laboratory, Mauna Loa Observatory,
  annual means (Keeling Curve), 1959–present.
  <https://gml.noaa.gov/ccgg/trends/data.html>

```python
# 1. NASA GISTEMP v4: global annual mean temperature anomalies
gistemp_url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

gistemp_raw = pd.read_csv(
    gistemp_url,
    skiprows=1,
    na_values=["****", "***", ""]
)

temp_df = (
    gistemp_raw
    .rename(columns={"Year": "year", "J-D": "temp_anomaly"})
    [["year", "temp_anomaly"]]
)

temp_df["year"] = pd.to_numeric(temp_df["year"], errors="coerce")
temp_df["temp_anomaly"] = pd.to_numeric(temp_df["temp_anomaly"], errors="coerce")
temp_df = temp_df.dropna()

temp_df["year"] = temp_df["year"].astype(int)

# 2. NOAA GML: Mauna Loa annual mean CO2
co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"

co2_raw = pd.read_csv(
    co2_url,
    comment="#",
    names=["year", "co2_ppm", "uncertainty"]
)

co2_raw["year"] = pd.to_numeric(co2_raw["year"], errors="coerce")
co2_raw["co2_ppm"] = pd.to_numeric(co2_raw["co2_ppm"], errors="coerce")
co2_raw = co2_raw.dropna()

co2_raw["year"] = co2_raw["year"].astype(int)

# 3. Join on overlapping years
climate_df = temp_df.merge(
    co2_raw[["year", "co2_ppm"]],
    on="year",
    how="inner"
)

climate_df.head()
```

### Time-series plots

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    climate_df["year"],
    climate_df["temp_anomaly"],
    linewidth=2
)
axes[0].axhline(0, linestyle="--", color="gray")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Temperature anomaly (C)")
axes[0].set_title("Global mean temperature anomaly\nNASA GISTEMP v4")

axes[1].plot(
    climate_df["year"],
    climate_df["co2_ppm"],
    linewidth=2
)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("CO2 (ppm)")
axes[1].set_title("Atmospheric CO2, Mauna Loa\nNOAA GML")

plt.tight_layout()
plt.show()
```

Both series trend upward over time. Because they share a common trend, a regression of temperature on CO₂ will look impressive on the surface — but the residuals will tell a different story.

### Fitting the regression

```python
fit_climate = smf.ols(
    "temp_anomaly ~ co2_ppm",
    data=climate_df
).fit()

fit_climate.summary()
```

The $R^2$ is high and the coefficient on CO₂ is highly significant. This looks like a strong result — but we need to check the residuals.

### Residual diagnostics: the autocorrelation problem

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    climate_df["year"],
    fit_climate.resid,
    linewidth=2
)
axes[0].axhline(0, linestyle="--", color="gray")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Residual")
axes[0].set_title("OLS residuals over time")

plot_acf(
    fit_climate.resid,
    ax=axes[1],
    title="ACF of OLS residuals"
)

plt.tight_layout()
plt.show()
```

**What to notice:**

* The residual plot is not random noise — it has a clear wave-like structure indicating that consecutive residuals are related (positive autocorrelation).
* The ACF confirms this: several lags are well outside the confidence bands.
* This violates the OLS independence assumption, so standard errors and p-values are unreliable — even though the regression *appears* to fit well.

### Why this matters

```python
dw = durbin_watson(fit_climate.resid)

dw
```

The Durbin-Watson statistic near 0 (versus the null of 2) confirms strong positive autocorrelation. Key takeaway: a high $R^2$ and a significant coefficient are not sufficient. You must inspect the residuals over time.

---

## Features of Time Series

The R version uses several datasets from the `fpp3` ecosystem. In this Python version, we use local files when available and otherwise create simple teaching datasets with the same time-series features.

```python
def make_quarterly_dates(start="1992-01-01", periods=76):
    return pd.date_range(start=start, periods=periods, freq="QS")
```

```python
# Trend: Australian GDP-style data
years = np.arange(1960, 2018)
gdp = 0.25e12 * np.exp(0.035 * (years - years[0])) + np.random.normal(0, 0.03e12, len(years))

aus_gdp = pd.DataFrame({
    "Year": years,
    "GDP": gdp
})

# Seasonality: Australian quarterly beer-style production
quarters = make_quarterly_dates("1992-01-01", 76)
season_pattern = np.array([420, 380, 400, 500])
beer_values = np.tile(season_pattern, int(np.ceil(len(quarters) / 4)))[:len(quarters)]
beer_values = beer_values + np.linspace(30, -40, len(quarters)) + np.random.normal(0, 15, len(quarters))

beer = pd.DataFrame({
    "Quarter": quarters,
    "Beer": beer_values
})

# Cycles: lynx-style pelt cycle
cycle_years = np.arange(1845, 1936)
lynx = 3500 + 2500 * np.sin(2 * np.pi * (cycle_years - 1845) / 10) + np.random.normal(0, 700, len(cycle_years))
lynx = np.maximum(lynx, 200)

pelt = pd.DataFrame({
    "Year": cycle_years,
    "Lynx": lynx
})

# Noise
np.random.seed(2026)
noise_ts = pd.DataFrame({
    "t": np.arange(1, 121),
    "value": np.random.normal(size=120)
})
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(aus_gdp["Year"], aus_gdp["GDP"] / 1e12)
axes[0, 0].set_title("Trend")
axes[0, 0].set_ylabel("GDP (trillions USD)")

axes[0, 1].plot(beer["Quarter"], beer["Beer"])
axes[0, 1].set_title("Seasonality")
axes[0, 1].set_ylabel("Megalitres")

axes[1, 0].plot(pelt["Year"], pelt["Lynx"])
axes[1, 0].set_title("Cycles")
axes[1, 0].set_ylabel("Number of pelts")

axes[1, 1].plot(noise_ts["t"], noise_ts["value"])
axes[1, 1].axhline(0, linestyle="--", color="gray")
axes[1, 1].set_title("Noise")
axes[1, 1].set_ylabel("Value")

plt.tight_layout()
plt.show()
```

### The Four Features

**Trend (top left).** The GDP series has a persistent upward movement across decades. That sustained long-run direction is what we mean by a trend.

**Seasonality (top right).** Beer production has a repeating quarterly pattern. In Australia, Q4 often rises because December is Australian summer. The pattern repeats at exactly the same calendar period every year — that fixed, calendar-locked periodicity is what distinguishes seasonality from cycles.

**Cycles (bottom left).** Lynx pelt counts oscillate roughly every 9–11 years, driven by the hare-lynx predator-prey dynamic. Unlike seasonal patterns, cycles do not have a fixed frequency — the period and amplitude vary from one cycle to the next, and they are not locked to the calendar.

**Noise (bottom right).** By construction there is no pattern here. Each value is an independent draw from $N(0,1)$. After a good model is fit to a real series, the residuals should look like this panel.

### Confirming noise with an ACF

A quick way to demonstrate that the white-noise panel has no structure is to check its autocorrelation function (ACF). All spikes should fall within the blue confidence bands.

```python
plot_acf(noise_ts["value"], lags=30)
plt.title("ACF of white noise — no significant autocorrelation")
plt.show()
```

Compare this to the ACF of the beer series, which has strong spikes at lags 4, 8, 12, etc. This is the signature of quarterly seasonality:

```python
plot_acf(beer["Beer"], lags=24)
plt.title("ACF of beer production — strong seasonal autocorrelation")
plt.show()
```

---

## Forecasting

```python
recent_beer = beer.copy()

train = recent_beer[recent_beer["Quarter"].dt.year <= 2007]
test = recent_beer[recent_beer["Quarter"].dt.year > 2007]

plt.figure(figsize=(10, 4))
plt.plot(train["Quarter"], train["Beer"], label="Training data")
plt.plot(test["Quarter"], test["Beer"], label="Future test data", color="gray")
plt.title("Hide the future first")
plt.legend()
plt.show()
```

## Exponential Smoothing

### Motivating the idea: should old data count as much as newer data?

A naive approach is the historical average:

$$\hat{y}_{t+1} = \frac{y_1 + y_2 + \cdots + y_t}{t}$$

For temperature or many economic series, an observation from three years ago probably should not count as much as yesterday's. This motivates giving more weight to recent
observations, which is the key idea behind exponential smoothing.

### Exponentially weighted averages

Exponential smoothing weights observations so that importance decays geometrically with age:

$$\hat{y}_{t+1} = \alpha y_t + \alpha(1-\alpha)y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots$$

The recursive (state-space) form makes this tractable to compute:

$$\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}$$

Here $\ell_t$ is the estimated **level** at time $t$. In plain language:

> New estimate = weighted average of what we just observed and what we previously
> believed.

The smoothing parameter $\alpha \in (0, 1)$ controls the tradeoff:

| Value of $\alpha$ | Behavior | Interpretation |
|---|---|---|
| Small (~0.1) | Very smooth | Slow to adapt |
| Medium (~0.5) | Balanced | Moderate adaptation |
| Large (~0.9) | Wiggly | Responds quickly |

### Manual demonstration of $\alpha$

We write the recursive update by hand so we can compare three smoothers side-by-side.

```python
def ses_level(y, alpha):
    y = np.asarray(y)
    level = np.zeros(len(y))
    level[0] = y[0]

    for t in range(1, len(y)):
        level[t] = alpha * y[t] + (1 - alpha) * level[t - 1]

    return level
```

Subset Australian beer production to a clean 16-year window:

```python
beer_window = beer[
    (beer["Quarter"] >= "1990-01-01") &
    (beer["Quarter"] <= "2005-12-31")
].copy()

beer_window.head()
```

### Plot three smoothers

Apply $\alpha \in \{0.10, 0.50, 0.90\}$ and compare visually:

```python
beer_smooth = beer_window.copy()

beer_smooth["alpha_10"] = ses_level(beer_smooth["Beer"], 0.10)
beer_smooth["alpha_50"] = ses_level(beer_smooth["Beer"], 0.50)
beer_smooth["alpha_90"] = ses_level(beer_smooth["Beer"], 0.90)

plt.figure(figsize=(10, 5))
plt.plot(beer_smooth["Quarter"], beer_smooth["Beer"], color="gray", label="Observed")
plt.plot(beer_smooth["Quarter"], beer_smooth["alpha_10"], label="alpha = 0.10 (slow)")
plt.plot(beer_smooth["Quarter"], beer_smooth["alpha_50"], label="alpha = 0.50 (balanced)")
plt.plot(beer_smooth["Quarter"], beer_smooth["alpha_90"], label="alpha = 0.90 (fast)")
plt.ylabel("Beer production (megalitres)")
plt.title("Australian beer production with three exponential smoothers")
plt.legend()
plt.show()
```

**Discussion:** Notice that $\alpha = 0.10$ traces the overall trend almost like a horizontal average, while $\alpha = 0.90$ tracks the quarterly spikes closely.

### Fit exponential smoothing with statsmodels

`ExponentialSmoothing()` fits members of the exponential smoothing family.

```python
beer_series = beer_window.set_index("Quarter")["Beer"]
beer_series = beer_series.asfreq("QS")

fit_ets = ExponentialSmoothing(
    beer_series,
    trend=None,
    seasonal="add",
    seasonal_periods=4
).fit()

fit_ets.summary()
```

The output shows which trend and seasonal structure was selected by the model specification.

### Forecast with exponential smoothing

```python
ets_forecast = fit_ets.forecast(8)

plt.figure(figsize=(10, 5))
plt.plot(beer_series.index, beer_series, label="Observed")
plt.plot(ets_forecast.index, ets_forecast, label="ETS forecast")
plt.ylabel("Beer production (megalitres)")
plt.title("ETS forecast: Australian beer production")
plt.legend()
plt.show()
```

The shaded bands are 80 % and 95 % prediction intervals. Notice that the intervals widen as the forecast horizon grows because uncertainty accumulates over time.

In this simplified Python example, we have plotted point forecasts only. Prediction intervals require additional steps or simulation.

### Trend and seasonality extensions

| Pattern in the data | Common method |
|---|---|
| No trend, no seasonality | Simple exponential smoothing (SES) |
| Trend only | Holt's linear method |
| Trend + seasonality | Holt-Winters method |

Exponential smoothing is a *family* of models. `ETS()` picks the right family member automatically, but you can also specify one explicitly:

```python
# Additive Holt-Winters-style model
holt_winters_fit = ExponentialSmoothing(
    beer_series,
    trend="add",
    seasonal="add",
    seasonal_periods=4
).fit()

holt_winters_fit.summary()
```

---

## Another method: Autoregressive Integrated Moving Averages (ARIMA)

### Autoregression

An AR(1) model predicts the next value from the previous one:

$$Y_t = \phi Y_{t-1} + \varepsilon_t$$

The contrast with exponential smoothing: ETS smooths past observations; AR models
explicitly parametrize *dependence* on lagged values.

### What ARIMA combines

ARIMA(p, d, q) packages three ideas:

* **AR(p):** autoregression — use the last $p$ values
* **I(d):** integration — difference $d$ times to remove trend
* **MA(q):** moving-average errors — use the last $q$ shocks

For this bootcamp, the goal is not to explain every parameter but to demonstrate the workflow.

### A minimal ARIMA example

```python
fit_arima = ARIMA(
    beer_series,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 4)
).fit()

fit_arima.summary()
```

```python
arima_forecast = fit_arima.forecast(8)

plt.figure(figsize=(10, 5))
plt.plot(beer_series.index, beer_series, label="Observed")
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA forecast")
plt.ylabel("Beer production (megalitres)")
plt.title("ARIMA forecast: Australian beer production")
plt.legend()
plt.show()
```

### Compare ETS and ARIMA side-by-side

```python
plt.figure(figsize=(10, 5))
plt.plot(beer_series.index, beer_series, label="Observed")
plt.plot(ets_forecast.index, ets_forecast, label="ETS")
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA")
plt.ylabel("Beer production (megalitres)")
plt.title("ETS vs ARIMA forecasts")
plt.legend()
plt.show()
```

Accuracy on the training data:

```python
ets_fitted = fit_ets.fittedvalues
arima_fitted = fit_arima.fittedvalues

def accuracy(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true).align(pd.Series(y_pred), join="inner")
    errors = y_true - y_pred
    return pd.Series({
        "RMSE": np.sqrt(np.mean(errors**2)),
        "MAE": np.mean(np.abs(errors)),
        "MAPE": np.mean(np.abs(errors / y_true)) * 100
    })

pd.DataFrame({
    "ETS": accuracy(beer_series, ets_fitted),
    "ARIMA": accuracy(beer_series, arima_fitted)
}).T
```

---

## And now for something completely different

## Bootstrap

### The idea

Given a sample $x_1, \ldots, x_n$, the bootstrap constructs an approximate sampling distribution for any statistic by:

1. Treating the sample as a stand-in for the population.
2. Drawing many resamples **with replacement** of size $n$.
3. Recomputing the statistic on each resample.

The spread of those recomputed values estimates uncertainty.

### Bootstrap a mean

We use the `airquality` dataset (daily air quality measurements in New York, 1973).

```python
try:
    airquality = sm.datasets.get_rdataset("airquality", "datasets").data
except Exception:
    # Fallback approximation if the R dataset cannot be downloaded
    np.random.seed(123)
    airquality = pd.DataFrame({
        "Ozone": np.random.gamma(shape=2, scale=20, size=150)
    })
    airquality.loc[np.random.choice(airquality.index, 35, replace=False), "Ozone"] = np.nan

airquality["Ozone"].describe()
```

```python
np.random.seed(123)

x = airquality["Ozone"].dropna().to_numpy()

boot_means = []

for _ in range(2000):
    x_star = np.random.choice(x, size=len(x), replace=True)
    boot_means.append(np.mean(x_star))

boot_means = np.array(boot_means)

np.quantile(boot_means, [0.025, 0.975])
```

Compare this to the classical t-interval:

```python
stats.t.interval(
    confidence=0.95,
    df=len(x) - 1,
    loc=np.mean(x),
    scale=stats.sem(x)
)
```

For a reasonably symmetric distribution, the two approaches agree closely.

### Visualize the bootstrap distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(boot_means, bins=30, edgecolor="white")
plt.axvline(np.quantile(boot_means, 0.025), linestyle="--")
plt.axvline(np.quantile(boot_means, 0.975), linestyle="--")
plt.xlabel("Bootstrap sample mean (ppb)")
plt.ylabel("Count")
plt.title("Bootstrap sampling distribution of mean ozone")
plt.show()
```

The bell shape of the bootstrap distribution reflects the central limit theorem at
work — and we obtained it without any distributional assumptions.

### Bootstrap a different statistic: the median

The bootstrap shines when the analytical sampling distribution is hard to derive.

```python
np.random.seed(42)

boot_medians = []

for _ in range(2000):
    x_star = np.random.choice(x, size=len(x), replace=True)
    boot_medians.append(np.median(x_star))

boot_medians = np.array(boot_medians)

np.quantile(boot_medians, [0.025, 0.975])
```

```python
plt.figure(figsize=(8, 5))
plt.hist(boot_medians, bins=25, edgecolor="white")
plt.axvline(np.quantile(boot_medians, 0.025), linestyle="--")
plt.axvline(np.quantile(boot_medians, 0.975), linestyle="--")
plt.xlabel("Bootstrap sample median (ppb)")
plt.ylabel("Count")
plt.title("Bootstrap sampling distribution of median ozone")
plt.show()
```

---

## Local Linear Regression Smoothing (LOESS)

### The idea

Ordinary linear regression fits one global line. Local regression (LOESS / LOWESS) fits many small weighted regressions, one for each $x_0$:

1. Give observations near $x_0$ high weight (using a kernel function).
2. Fit a small weighted regression locally.
3. Read off the predicted value at $x_0$.
4. Move $x_0$ across the range of $x$ to trace out a smooth curve.

The **span** (bandwidth) controls how wide the local neighborhood is — and therefore
how smooth the resulting curve is.

### LOESS in Python with `mtcars`

```python
try:
    mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data.reset_index()
except Exception:
    mtcars = pd.DataFrame({
        "wt": [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78],
        "mpg": [21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2],
        "cyl": [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8]
    })

mtcars.head()
```

```python
loess_default = lowess(
    mtcars["mpg"],
    mtcars["wt"],
    frac=0.75,
    return_sorted=True
)

plt.figure(figsize=(8, 5))
plt.scatter(mtcars["wt"], mtcars["mpg"], color="gray")
plt.plot(loess_default[:, 0], loess_default[:, 1])
plt.xlabel("Weight (1000 lbs)")
plt.ylabel("Miles per gallon")
plt.title("Local regression smoother (default span = 0.75)")
plt.show()
```

### Effect of the span (bandwidth)

```python
spans = [0.2, 0.5, 0.75]

plt.figure(figsize=(8, 5))
plt.scatter(mtcars["wt"], mtcars["mpg"], color="gray")

for span in spans:
    fitted = lowess(
        mtcars["mpg"],
        mtcars["wt"],
        frac=span,
        return_sorted=True
    )
    plt.plot(fitted[:, 0], fitted[:, 1], label=f"span = {span}")

plt.xlabel("Weight (1000 lbs)")
plt.ylabel("Miles per gallon")
plt.title("LOWESS curves with three different spans")
plt.legend()
plt.show()
```

A small span (0.2) chases every point; a large span (0.75) is smoother but may miss real curvature. Use LOESS as an **exploratory** tool, not automatic proof of a relationship.

---

## Kernel Density Estimation

### The idea

A histogram estimates a distribution's shape, but the visual result depends heavily on the chosen bin width and starting edge. Kernel density estimation (KDE) places a smooth "bump" (kernel) at each observation and sums the bumps:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)$$

KDE is a **smooth version of a histogram**. The bandwidth $h$ plays the same role as bin width — it controls flexibility.

| Bandwidth choice | Result |
|---|---|
| Too small | Too wiggly — chases individual points |
| Reasonable | Reveals the true distribution shape |
| Too large | Over-smoothed — washes out real features |

### KDE in Python: histogram + density overlay

```python
mpg = mtcars["mpg"].to_numpy()

kde = gaussian_kde(mpg)
x_grid = np.linspace(mpg.min() - 5, mpg.max() + 5, 512)

plt.figure(figsize=(8, 5))
plt.hist(mpg, bins=10, density=True, alpha=0.6, edgecolor="white")
plt.plot(x_grid, kde(x_grid))
plt.xlabel("Miles per gallon")
plt.ylabel("Density")
plt.title("Histogram plus kernel density estimate (mtcars)")
plt.show()
```

### Effect of bandwidth

```python
bandwidths = [0.3, 0.7, 1.5]

plt.figure(figsize=(8, 5))

for bw in bandwidths:
    kde = gaussian_kde(mpg)
    kde.set_bandwidth(kde.factor * bw)
    plt.plot(x_grid, kde(x_grid), label=f"bandwidth factor = {bw}")

plt.xlabel("Miles per gallon")
plt.ylabel("Density")
plt.title("KDE with three bandwidth choices")
plt.legend()
plt.show()
```

The default bandwidth is usually a good starting point, but always inspect the curve for over- or under-smoothing.

### KDE for comparing groups

```python
plt.figure(figsize=(8, 5))

for cyl, subset in mtcars.groupby("cyl"):
    values = subset["mpg"].to_numpy()
    if len(values) > 1:
        kde = gaussian_kde(values)
        grid = np.linspace(mpg.min() - 5, mpg.max() + 5, 512)
        plt.plot(grid, kde(grid), label=f"{cyl} cyl")

plt.xlabel("Miles per gallon")
plt.ylabel("Density")
plt.title("MPG distribution by number of cylinders")
plt.legend()
plt.show()
```

---

## Putting It All Together

Work with the `aus_retail` dataset (monthly Australian retail turnover by industry).

In this Python adaptation, we use a teaching dataset that mimics monthly cafe and restaurant turnover.

```python
np.random.seed(2026)

dates = pd.date_range("2000-01-01", "2018-12-01", freq="MS")
trend = np.linspace(600, 1200, len(dates))
season = 80 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
shock = np.where((dates >= "2008-01-01") & (dates <= "2009-12-01"), -70, 0)
noise = np.random.normal(0, 35, len(dates))

cafe = pd.DataFrame({
    "Month": dates,
    "Turnover": trend + season + shock + noise
})

plt.figure(figsize=(10, 4))
plt.plot(cafe["Month"], cafe["Turnover"])
plt.ylabel("Turnover (AUD million)")
plt.title("Victorian cafes and restaurants: monthly turnover")
plt.show()
```

**Question 1 — What patterns do you see?**

```python
from statsmodels.tsa.seasonal import STL

cafe_series = cafe.set_index("Month")["Turnover"].asfreq("MS")

stl = STL(cafe_series, period=12).fit()

stl.plot()
plt.show()
```

**Question 2 — Which forecast would you trust?**

```python
train = cafe[cafe["Month"] <= "2016-12-01"].copy()
test = cafe[cafe["Month"] >= "2017-01-01"].copy()

train_series = train.set_index("Month")["Turnover"].asfreq("MS")
test_series = test.set_index("Month")["Turnover"].asfreq("MS")

fit_ets_cafe = ExponentialSmoothing(
    train_series,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

fit_arima_cafe = ARIMA(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12)
).fit()

# Seasonal naive benchmark: repeat the value from 12 months ago
snaive_forecast = train_series.iloc[-12:].to_numpy()
snaive_forecast = np.resize(snaive_forecast, len(test_series))

ets_fc = fit_ets_cafe.forecast(len(test_series))
arima_fc = fit_arima_cafe.forecast(len(test_series))

plt.figure(figsize=(10, 5))
plt.plot(cafe["Month"], cafe["Turnover"], label="Observed")
plt.plot(test_series.index, ets_fc, label="ETS")
plt.plot(test_series.index, arima_fc, label="ARIMA")
plt.plot(test_series.index, snaive_forecast, label="Seasonal naive")
plt.plot(test_series.index, test_series, color="black", linewidth=2, label="Test data")
plt.ylabel("Turnover (AUD million)")
plt.title("Three-model comparison with held-out 2017 test data")
plt.legend()
plt.show()

def forecast_accuracy(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    errors = actual - predicted

    return pd.Series({
        "RMSE": np.sqrt(np.mean(errors**2)),
        "MAE": np.mean(np.abs(errors)),
        "MAPE": np.mean(np.abs(errors / actual)) * 100
    })

pd.DataFrame({
    "ETS": forecast_accuracy(test_series, ets_fc),
    "ARIMA": forecast_accuracy(test_series, arima_fc),
    "Naive": forecast_accuracy(test_series, snaive_forecast)
}).T.sort_values("RMSE")
```

**Question 3 — How would you quantify uncertainty if this were not a time series?**

```python
np.random.seed(2026)

y = cafe["Turnover"].to_numpy()

boot_turnover = []

for _ in range(2000):
    boot_turnover.append(
        np.mean(np.random.choice(y, size=len(y), replace=True))
    )

boot_turnover = np.array(boot_turnover)

print("Bootstrap 95% CI for mean monthly turnover:")
np.quantile(boot_turnover, [0.025, 0.975])
```

```python
kde = gaussian_kde(y)
x_grid = np.linspace(y.min() - 50, y.max() + 50, 512)

plt.figure(figsize=(8, 5))
plt.hist(y, bins=20, density=True, alpha=0.6, edgecolor="white")
plt.plot(x_grid, kde(x_grid))
plt.xlabel("Monthly turnover (AUD million)")
plt.ylabel("Density")
plt.title("Distribution of monthly turnover (Victorian cafes)")
plt.show()
```

---

## Key Takeaways

1. Time order matters — nearby observations are often related.
2. Ordinary regression can be misleading for autocorrelated or trending data.
3. Forecasting models should be evaluated on **held-out future data**.
4. Start with simple benchmarks (seasonal naïve, mean).
5. Try exponential smoothing — it handles trend and seasonality automatically.
6. Treat ARIMA as a next step once you understand autocorrelation.
7. Use bootstrap, LOESS, and KDE as general data-analysis tools regardless of whether the data are a time series.
8. When data are a time series, there are versions of bootstrap that will preserve the sequential order of the series.
