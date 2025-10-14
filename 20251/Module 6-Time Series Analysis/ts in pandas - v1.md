# Crash Course: Time Series Analysis and Trend Discovery with Pandas

## Learning Objectives

By the end of this session, students should be able to:

1. Load and manipulate time series data using Pandas.
2. Handle datetime indices and extract time-based features.
3. Resample, smooth, and visualize time series data.
4. Detect trends using moving averages and differencing.
5. Perform basic decomposition into trend, seasonality, and residuals.

---

## 1. Understanding Time Series Data

A **time series** is a sequence of observations recorded over time. Examples include stock prices, daily sales, temperature readings, and website traffic.

In Pandas, time series data is usually represented with a **datetime index**.

```python
import pandas as pd

# Example: loading a CSV with a Date column
df = pd.read_csv('sales.csv', parse_dates=['Date'], index_col='Date')
print(df.head())
```

* `parse_dates` converts string dates to datetime objects.
* Setting the `Date` as index enables time-based operations such as resampling and filtering.

---

## 2. Exploring the Datetime Index

Once the index is of type datetime, it can be used to extract time-based attributes.

```python
df.index.year
df.index.month
df.index.day_name()

df['2024']                  # filter by year
df['2024-06':'2024-08']     # filter by date range
```

Example plot:

```python
import matplotlib.pyplot as plt

df['Sales'].plot(figsize=(10, 4), title='Daily Sales')
plt.show()
```

---

## 3. Resampling and Aggregation

Resampling changes the frequency of observations, such as converting daily data to monthly totals or averages.

```python
# Convert daily data to monthly totals
monthly = df['Sales'].resample('M').sum()

# Convert to monthly average
monthly_mean = df['Sales'].resample('M').mean()

monthly_mean.plot(title='Monthly Average Sales', figsize=(10, 4))
```

Common frequency codes:

* `'D'` – Day
* `'W'` – Week
* `'M'` – Month
* `'Q'` – Quarter
* `'Y'` – Year

---

## 4. Moving Averages and Smoothing

To discover underlying trends, data can be smoothed using moving averages.

```python
df['RollingMean_7'] = df['Sales'].rolling(window=7).mean()
df[['Sales', 'RollingMean_7']].plot(title='7-Day Moving Average', figsize=(10, 4))
```

* A short window captures local fluctuations.
* A long window reveals long-term trends.

---

## 5. Detecting Trends Using Differencing

Differencing helps remove trends or seasonality to make the series stationary.

```python
df['Diff'] = df['Sales'].diff()
df[['Sales', 'Diff']].plot(subplots=True, figsize=(10, 6), title=['Original', 'Differenced'])
```

This allows comparison between the raw data and its changes over time.

---

## 6. Trend and Seasonality Decomposition

To separate the trend, seasonal, and residual components, use the `seasonal_decompose` function from `statsmodels`.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Sales'], model='additive', period=30)
result.plot()
plt.show()
```

Decomposition components:

* **Trend:** Long-term direction of the data
* **Seasonality:** Repeating patterns or cycles
* **Residual:** Random fluctuations

---

## 7. Correlation and Lag Analysis

Lag analysis helps identify relationships over time by comparing values at different time steps.

```python
df['Lag1'] = df['Sales'].shift(1)
df[['Sales', 'Lag1']].corr()
```

Autocorrelation can help detect seasonality or periodic patterns.

```python
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
```

---

## 8. Case Study: Airline Passengers Dataset

A classic example of time series analysis is the airline passengers dataset.

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
airline = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
airline.plot(title='Monthly Airline Passengers', figsize=(10, 4))
```

Detecting the long-term trend using a 12-month moving average:

```python
airline['MA12'] = airline['Passengers'].rolling(window=12).mean()
airline[['Passengers', 'MA12']].plot(title='12-Month Moving Average', figsize=(10, 4))
```

---

## 9. Simple Trend Forecasting (Optional)

A linear trend can be estimated using linear regression.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

airline['t'] = np.arange(len(airline))
X = airline[['t']]
y = airline['Passengers']

model = LinearRegression().fit(X, y)
airline['Trend'] = model.predict(X)

airline[['Passengers', 'Trend']].plot(title='Trend Line', figsize=(10, 4))
```

This illustrates a simple way to estimate and visualize the trend component.

---

## 10. Summary of Key Techniques

| Technique                | Purpose                        | Function                  |
| ------------------------ | ------------------------------ | ------------------------- |
| `resample()`             | Change data frequency          | `df.resample('M').mean()` |
| `rolling()`              | Smooth data                    | `df.rolling(7).mean()`    |
| `diff()`                 | Remove trend                   | `df.diff()`               |
| `shift()`                | Create lags                    | `df.shift(1)`             |
| `autocorrelation_plot()` | Detect periodicity             | from `pandas.plotting`    |
| `seasonal_decompose()`   | Separate trend and seasonality | from `statsmodels`        |

---

## Suggested Exercise

1. Load any time series dataset (for example, COVID-19 cases, stock prices, or weather data).
2. Resample the data to a monthly frequency.
3. Plot rolling averages and perform seasonal decomposition.
4. Identify and describe the main trend observed in the data.
5. Submit a short Jupyter Notebook summarizing the findings.

---
