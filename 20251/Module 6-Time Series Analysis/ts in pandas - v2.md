## Pandas for Time Series Analysis Crash Course

-----

### 1\. The Time Series Index

The foundation of time series analysis in Pandas is the **DatetimeIndex**.

  * **Creating a DatetimeIndex:** Your time-based data must be correctly indexed. Use the `pd.to_datetime()` function to convert a column of strings or numbers into datetime objects, and then set it as the DataFrame's index using `df.set_index()`.

    ```python
    import pandas as pd
    # Assuming 'Date_Column' is a string or integer column
    df['Date'] = pd.to_datetime(df['Date_Column'])
    df = df.set_index('Date')
    ```

  * **Frequency and Resampling:** Time series data often needs to be standardized to a specific frequency (e.g., from daily to monthly). **Resampling** is the process of changing that frequency.

    | Alias | Description | Example |
    | :---: | :---: | :---: |
    | $\text{'B'}$ | Business day frequency | $\text{df.resample('B').mean()}$ |
    | $\text{'W'}$ | Weekly frequency | $\text{df.resample('W').sum()}$ |
    | $\text{'M'}$ | Month end frequency | $\text{df.resample('M').max()}$ |
    | $\text{'Q'}$ | Quarter end frequency | $\text{df.resample('Q').mean()}$ |

    Use an aggregation function (like $\text{.mean()}$, $\text{.sum()}$, $\text{.min()}$, or $\text{.max()}$) after $\text{resample()}$.

    ```python
    # Aggregate daily data to monthly averages
    monthly_data = df['Value'].resample('M').mean()
    ```

-----

### 2\. Discovering Trends with Pandas

A **trend** in a time series is a long-term movement in a particular direction (upward, downward, or horizontal), ignoring short-term fluctuations and seasonality.

#### A. Rolling Statistics (Moving Averages)

The simplest and most common method for smoothing a time series to reveal the underlying trend is using **Rolling or Moving Averages** ($\text{MA}$).

  * **Concept:** An $\text{MA}$ is calculated by taking the average of a specific number of preceding data points. This smooths out the "noise" (random fluctuations) and makes the trend more apparent. A longer window size results in a smoother line, highlighting the long-term trend more clearly.

  * **Pandas Implementation:** Use the $\text{df.rolling(window=N)}$ method, where $\text{N}$ is the number of periods to include in the calculation.

    ```python
    # Calculate a 30-day (or 30-period) Moving Average
    df['30D_MA'] = df['Value_Column'].rolling(window=30).mean()

    # To visualize the trend, plot the original data and the MA on the same chart.
    df[['Value_Column', '30D_MA']].plot(figsize=(10, 5))
    ```

#### B. Expanding Window Statistics (Cumulative Analysis)

**Expanding window statistics** show how a metric behaves over all time up to the current point. This is useful for analyzing cumulative effects or long-term growth.

  * **Concept:** For each point in the series, the statistic (e.g., mean, sum) is calculated using *all* data points from the start of the series up to the current point.

  * **Pandas Implementation:** Use the $\text{df.expanding()}$ method.

    ```python
    # Calculate the cumulative mean from the start of the series
    df['Cumulative_Mean'] = df['Value_Column'].expanding().mean()
    ```

#### C. Differencing (Removing Trend and Seasonality)

While not a direct trend discovery tool, **differencing** is a crucial technique used to **isolate** the trend and seasonality for modeling purposes.

  * **Concept:** Differencing calculates the change from one period to the next. The first difference ($d=1$) removes the linear trend. If the trend is non-linear, you may need a second difference ($d=2$).

  * **Pandas Implementation:** Use the $\text{df.diff(periods=N)}$ method.

    ```python
    # Calculate the first difference (change from one period to the next)
    df['Difference'] = df['Value_Column'].diff(periods=1)
    # The original trend is removed in the 'Difference' series.
    ```

#### D. Decomposition

**Time Series Decomposition** breaks down the series into its four fundamental components:

1.  **Trend ($\text{T}$):** The long-term direction.
2.  **Seasonality ($\text{S}$):** Repeating cycles at fixed intervals (e.g., yearly, weekly).
3.  **Cyclical ($\text{C}$):** Fluctuations that are not fixed periods (often associated with business cycles).
4.  **Residual/Error ($\text{E}$):** The unpredictable part.

Pandas does not have a built-in decomposition function, but its statistical companion, **Statsmodels**, is standard for this.

  * **Model Types:**

      * **Additive:** $\text{Y(t) = T(t) + S(t) + E(t)}$ (When the magnitude of seasonality is constant).
      * **Multiplicative:** $\text{Y(t) = T(t) * S(t) * E(t)}$ (When the magnitude of seasonality increases with the trend).

    <!-- end list -->

    ```python
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Assuming 'df' is a series with a DatetimeIndex
    decomposition = seasonal_decompose(df['Value_Column'], model='additive', period=12) # Use period based on known seasonality (e.g., 12 for monthly data, 7 for daily)

    # The trend component is directly accessible:
    trend = decomposition.trend
    # Plotting the trend component clearly reveals the long-term movement
    trend.plot(title='Extracted Trend Component')
    ```