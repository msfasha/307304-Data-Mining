# **Module 1: Time Series Fundamentals and Setup**

Welcome to the first module of our Section\! Before we jump into complex forecasting models, we need a solid understanding of what a time series is and how Python helps us handle this unique type of data.

## **1.1 What is a Time Series?**

A time series is simply a sequence of data points indexed, ordered, or graphed in time. Unlike typical regression problems where observations are independent, time series data is characterized by *temporal dependence*, meaning the value at time  is often related to the value at time .

### **Key Components of a Time Series**

Any observed time series data () can usually be broken down into three fundamental components. Understanding these components is the first step in successful analysis.

1. **Trend ():** The long-term direction of the data. This could be increasing (e.g., global population), decreasing (e.g., sales of CDs), or constant.  
2. **Seasonality ():** A regular, repeating pattern of predictable change. This occurs over a fixed period, such as daily, weekly, quarterly, or yearly.  
3. **Residual/Noise ():** The random, irregular fluctuations that remain after the trend and seasonal components have been removed. This is the unpredictable element.

### **Additive vs. Multiplicative Models**

How these components interact defines the type of time series model:

* Additive Model: The magnitude of the seasonal fluctuations does not depend on the level of the time series. This is often used when the variance remains roughly constant over time.

* Multiplicative Model: The magnitude of the seasonal fluctuations increases proportionally with the level of the time series (i.e., the seasonal peaks get larger as the overall trend increases).

## **1.2 Python Tools for Time Series**

Python has an incredibly powerful ecosystem for data science. For time series, we rely primarily on four core libraries:

* **Pandas:** This is the bedrock. Its DatetimeIndex structure is essential for time-indexed data. Functions like .resample() (to change frequency) and .shift() (to create lagged features) are cornerstones of time series preparation.  
* **NumPy:** Used for fast, vectorized numerical operations.  
* **Matplotlib/Seaborn:** Critical for visualizing data, decomposing components, and plotting forecasts.  
* **Statsmodels:** This library implements classic statistical tests (like the ADF test for stationarity) and traditional models (ARIMA, SARIMA).  
* **Scikit-learn (Sklearn):** While not exclusively a time series library, it provides useful tools for splitting data and calculating performance metrics (MSE, RMSE) and is necessary for applying modern ML models.

## **1.3 Loading and Indexing Time Series Data**

The most critical step in Pandas is ensuring your time column is correctly set as a DatetimeIndex. This enables powerful time-based operations.

### **Practical Example: Loading and Initial Plot**

We will simulate loading a hypothetical sales\_data.csv. Notice the key arguments: index\_col sets the 'Date' column as the row index, and parse\_dates=True ensures Pandas correctly interprets the strings in that column as datetime objects.

import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np

\# \--- Data Simulation (In a real scenario, this would be pd.read\_csv) \---  
\# Create a dummy time series for demonstration  
dates \= pd.date\_range(start='2020-01-01', periods=100, freq='W')  
np.random.seed(42)  
trend \= np.linspace(50, 150, 100\)  
seasonality \= 20 \* np.sin(np.linspace(0, 4 \* np.pi, 100))  
noise \= np.random.normal(0, 5, 100\)  
sales \= trend \+ seasonality \+ noise

data \= pd.DataFrame({'Sales': sales}, index=dates)  
data.index.name \= 'Date'  
\# \--- End of Simulation \---

print("--- Data Head (Notice the DatetimeIndex) \---")  
print(data.head())  
print("\\n--- Data Index Type \---")  
print(data.index)

\# Visualize the raw time series  
plt.figure(figsize=(12, 6))  
plt.plot(data\['Sales'\], color='\#0077b6', linewidth=2)  
plt.title('Raw Sales Time Series (Time-Based Indexing)', fontsize=16)  
plt.xlabel('Date')  
plt.ylabel('Sales Volume')  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()

\# Time-based slicing example  
print("\\n--- Sales in the year 2021 (Slicing) \---")  
data\_2021 \= data\['2021'\]  
print(data\_2021.tail())

\# Plotting the sliced data  
plt.figure(figsize=(8, 4))  
plt.plot(data\_2021\['Sales'\], color='\#fca311')  
plt.title('Sales Data for 2021', fontsize=14)  
plt.show()

## **Next Steps**

In **Module 2**, we will immediately apply the decomposition techniques learned here to better understand the trend and seasonality, and introduce the crucial concept of **stationarity**.

# **Module 2: Data Exploration and Preprocessing (EDA)**

The goal of exploratory data analysis (EDA) in time series is to identify and understand the underlying patterns—Trend and Seasonality—and prepare the data for statistical modeling.

## **Setup: Reloading Data**

To ensure this notebook section is fully runnable, we will re-initialize the simulated weekly sales data we created in Module 1\.

import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
from statsmodels.tsa.seasonal import seasonal\_decompose

\# \--- Data Simulation \---  
\# Weekly data over 100 weeks (\~2 years)  
dates \= pd.date\_range(start='2020-01-01', periods=100, freq='W')  
np.random.seed(42)  
trend \= np.linspace(50, 150, 100\) \# Increasing trend  
\# Simple annual seasonality (period 52 weeks in this weekly series)  
seasonality \= 20 \* np.sin(np.linspace(0, 4 \* np.pi, 100))   
noise \= np.random.normal(0, 5, 100\)  
sales \= trend \+ seasonality \+ noise

data \= pd.DataFrame({'Sales': sales}, index=dates)  
data.index.name \= 'Date'  
print(data.head())

## **2.1 Visualizing Components: Decomposition**

The seasonal\_decompose function from statsmodels is the quickest way to visualize the individual components of your time series. We need to specify the model type (additive or multiplicative) and the seasonal period.

* **Period:** Since our synthetic data is weekly (freq='W') and has an annual cycle, the period is  (weeks in a year). If it were monthly data with an annual cycle, the period would be .

\# Use a multiplicative model since the seasonality seems to scale with the trend.  
\# Period is set to 52 for annual cycle in weekly data.  
decomposition \= seasonal\_decompose(data\['Sales'\], model='multiplicative', period=52)

\# Plotting the decomposition  
fig, axes \= plt.subplots(4, 1, figsize=(12, 10), sharex=True)

decomposition.observed.plot(ax=axes\[0\], title='Observed (Raw Data)')  
decomposition.trend.plot(ax=axes\[1\], title='Trend Component')  
decomposition.seasonal.plot(ax=axes\[2\], title='Seasonal Component')  
decomposition.resid.plot(ax=axes\[3\], title='Residual Component')

plt.tight\_layout()  
plt.show()

\# 

## **2.2 Time Series Preprocessing: Resampling**

Often, the data frequency you receive is not the frequency you need for modeling. **Resampling** allows you to aggregate (down-sampling) or interpolate (up-sampling) data.

We use the Pandas .resample() method, followed by an aggregation function (like mean(), sum(), or first()).

| Alias | Description | Example Use Case |
| :---- | :---- | :---- |
| **D** | Daily | Daily stock prices |
| **W** | Weekly | Sales data aggregation |
| **M** | Month end | Monthly GDP |
| **Q** | Quarter end | Quarterly earnings |
| **A** | Year end | Annual revenue |

### **Example: Weekly to Monthly Resampling**

Let's convert our weekly sales data into monthly sales data using the mean of the values within each month.

\# Convert weekly data (W) to monthly data (M) using the mean value  
monthly\_data \= data.resample('M').mean()

print("--- Original Weekly Data Tail \---")  
print(data.tail())

print("\\n--- Resampled Monthly Data Tail \---")  
print(monthly\_data.tail())

\# Visualize the effect of resampling  
plt.figure(figsize=(12, 6))  
plt.plot(data\['Sales'\], label='Weekly (Original)', alpha=0.6)  
plt.plot(monthly\_data\['Sales'\], label='Monthly (Mean)', color='red', linewidth=2)  
plt.title('Effect of Resampling: Weekly vs. Monthly Mean')  
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()

## **2.3 The Core Concept: Stationarity**

### **Definition**

A time series is **stationary** if its statistical properties—mean, variance, and autocorrelation structure—do not change over time.

**Why is it important?** Many classical time series models (like ARIMA) assume the underlying data generating process is stationary. If a series is not stationary, its characteristics can only be studied for the specific time period it was observed.

### **Visual Check: Rolling Statistics**

We can visually check for stationarity by plotting the **rolling mean** and **rolling standard deviation**. If the series is stationary, both lines should be flat and close to constant.

\# Define window size (e.g., 12 periods/weeks)  
window \= 12 

\# Calculate rolling statistics  
roll\_mean \= data\['Sales'\].rolling(window=window).mean()  
roll\_std \= data\['Sales'\].rolling(window=window).std()

\# Plot rolling statistics  
plt.figure(figsize=(12, 6))  
plt.plot(data\['Sales'\], label='Original', color='\#0077b6', alpha=0.7)  
plt.plot(roll\_mean, label=f'Rolling Mean ({window} Wk)', color='red')  
plt.plot(roll\_std, label=f'Rolling Std ({window} Wk)', color='black')  
plt.title('Checking for Stationarity with Rolling Statistics')  
plt.legend()  
plt.show()

\# Observation: Since the red line (Rolling Mean) is clearly increasing,   
\# the data is NOT stationary (it has a trend).

### **Making a Series Stationary: Differencing**

The most common technique to achieve stationarity is differencing. This involves subtracting the previous observation (Yt−1​) from the current observation (Yt​):

This process removes the trend. If seasonality is present, you might use seasonal differencing (, where  is the seasonal period).

\# Apply first-order differencing (d=1)  
data\['Sales\_Differenced'\] \= data\['Sales'\].diff().dropna()

\# Plot the differenced series  
plt.figure(figsize=(12, 6))  
plt.plot(data\['Sales\_Differenced'\], color='darkgreen')  
plt.title('Sales Data After First-Order Differencing')  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()

\# Check rolling statistics on the differenced data  
roll\_mean\_diff \= data\['Sales\_Differenced'\].rolling(window=window).mean()

\# Notice how the differenced series appears to hover around zero (constant mean)  
plt.figure(figsize=(12, 6))  
plt.plot(data\['Sales\_Differenced'\], label='Differenced Series', alpha=0.7)  
plt.plot(roll\_mean\_diff, label=f'Rolling Mean ({window} Wk) \- Differenced', color='red')  
plt.title('Rolling Mean Check on Differenced Data')  
plt.legend()  
plt.show()

## **Next Steps**

Visually, the differenced series looks much more stationary. However, visual inspection is subjective. In **Module 3**, we will learn how to formally test for stationarity using the **Augmented Dickey-Fuller (ADF) Test** and then use the **ACF** and **PACF** plots to determine the parameters for our first statistical model: **ARIMA**.

# **Module 3: Statistical Modeling (ARIMA Framework)**

In Module 2, we visually confirmed that our sales data was non-stationary and applied differencing to stabilize the mean. In this module, we will perform a formal statistical test for stationarity and learn how to use the **Autocorrelation (ACF)** and **Partial Autocorrelation (PACF)** plots to select the parameters for our first model: **ARIMA**.

## **Setup: Data Initialization**

We'll re-create the data and the differenced series, as this is the input we need for our analysis.

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statsmodels.tsa.stattools import adfuller  
from statsmodels.graphics.tsaplots import plot\_acf, plot\_pacf  
from statsmodels.tsa.arima.model import ARIMA

\# \--- Data Simulation \---  
\# Weekly data over 100 weeks (\~2 years)  
dates \= pd.date\_range(start='2020-01-01', periods=100, freq='W')  
np.random.seed(42)  
trend \= np.linspace(50, 150, 100\)  
seasonality \= 20 \* np.sin(np.linspace(0, 4 \* np.pi, 100))   
noise \= np.random.normal(0, 5, 100\)  
sales \= trend \+ seasonality \+ noise

data \= pd.DataFrame({'Sales': sales}, index=dates)  
data.index.name \= 'Date'

\# Differencing (d=1) required for stationarity  
data\['Sales\_Differenced'\] \= data\['Sales'\].diff().dropna()  
print("Differenced Data Head:")  
print(data\['Sales\_Differenced'\].head())

## **3.1 Testing for Stationarity: The ADF Test**

While visual inspection with rolling statistics is helpful, we must use a formal statistical test. The most common is the **Augmented Dickey-Fuller (ADF) Test**.

### **ADF Test Hypotheses:**

* **Null Hypothesis ():** The time series has a unit root, meaning it is **non-stationary**.  
* **Alternative Hypothesis ():** The time series is stationary.

If the p-value is less than a chosen significance level (e.g., ), we reject the null hypothesis and conclude the series is stationary.

### **Example: Running ADF on Original and Differenced Data**

def check\_adfuller(series, name):  
    """Utility function to run and interpret the ADF test."""  
    print(f"--- Running ADF Test on {name} \---")  
    result \= adfuller(series)  
      
    print(f'ADF Statistic: {result\[0\]:.4f}')  
    print(f'P-value: {result\[1\]:.4f}')  
    print('Critical Values:')  
    for key, value in result\[4\].items():  
        print(f'  {key}: {value:.4f}')  
      
    if result\[1\] \<= 0.05:  
        print(f"\\nConclusion: Reject H0. The {name} series is likely stationary.")  
    else:  
        print(f"\\nConclusion: Fail to reject H0. The {name} series is non-stationary.")  
    print("-" \* 40\)

\# 1\. Test the original non-stationary series  
check\_adfuller(data\['Sales'\], "Original Sales")

\# 2\. Test the differenced series  
check\_adfuller(data\['Sales\_Differenced'\], "Differenced Sales")

**Observation:** The original series' p-value is high (failing to reject ), confirming non-stationarity. After differencing, the p-value is significantly low, confirming the series is now stationary. This means our  parameter in ARIMA is .

## **3.2 Autocorrelation and Partial Autocorrelation (ACF/PACF)**

To determine the remaining  and  parameters, we analyze the structure of the stationary (differenced) series using ACF and PACF plots.

| Plot | Acronym | Purpose (What it measures) | Determines ARIMA Parameter |
| :---- | :---- | :---- | :---- |
| **ACF** | Autocorrelation Function | Correlation between a series and its lagged values, including indirect effects from shorter lags. | **MA order ()** |
| **PACF** | Partial Autocorrelation Function | Correlation between a series and its lagged values, *excluding* indirect effects. | **AR order ()** |

The parameters  and  are typically chosen based on where the plots *cut off* (drop sharply to zero and stay there) or *taper off* (gradually decay).

### **Example: Plotting ACF and PACF on Differenced Data**

We plot these functions on the **stationary (differenced)** data.

\# Create the figure and axes  
fig, (ax1, ax2) \= plt.subplots(2, 1, figsize=(12, 8))

\# ACF Plot: Determines the MA (q) order  
plot\_acf(data\['Sales\_Differenced'\], lags=25, ax=ax1, title='Autocorrelation Function (ACF)')

\# PACF Plot: Determines the AR (p) order  
plot\_pacf(data\['Sales\_Differenced'\], lags=25, ax=ax2, title='Partial Autocorrelation Function (PACF)')

plt.tight\_layout()  
plt.show()

\# Interpretation:  
\# ACF: Shows a slow decay (tapering off) after lag 1\.  
\# PACF: Shows a sharp cut-off after lag 1, and then smaller spikes at lags 4 and 5\.  
\# Tentative Order based on the PACF cut-off: AR(1) or AR(5).

## **3.3 The ARIMA Model** 

The **A**utoregressive **I**ntegrated **M**oving **A**verage model combines these three concepts:

* **AR ():** The number of lagged observations included in the model (from PACF).  
* **I ():** The number of times the raw observations are differenced to achieve stationarity (from ADF test).  
* **MA ():** The size of the moving average window, or the number of lagged forecast errors included (from ACF).

The formula for an ARIMA(p,1,q) model (where d=1) is:

Where ΔYt​ is the differenced series, ϕ are the AR coefficients, θ are the MA coefficients, and ϵ are the error terms.

## **3.4 Practical ARIMA Implementation**

Based on our analysis (ADF gave , PACF suggests  or , ACF suggests  or ): let's try a simple  model first.

\# Fitting ARIMA(p=1, d=1, q=1)  
\# Note: The 'Sales' column (original non-stationary data) is passed to ARIMA.  
\# The 'd' parameter handles the differencing internally.

try:  
    model \= ARIMA(data\['Sales'\], order=(1, 1, 1))  
    model\_fit \= model.fit()

    print("--- ARIMA(1, 1, 1\) Model Summary \---")  
    print(model\_fit.summary())

    \# Plotting the model diagnostics (residuals)  
    model\_fit.plot\_diagnostics(figsize=(12, 8))  
    plt.show()

except Exception as e:  
    print(f"An error occurred during model fitting: {e}")  
    print("This may happen if the synthetic data structure is poorly suited for the chosen order.")

### **Interpreting the Summary**

* **:** Look for p-values near the AR and MA coefficients. If they are , the coefficients are significant.  
* **Ljung-Box Test (Prob(Q)):** Tests if the residuals are white noise (random). If this p-value is high (), the residuals are random, meaning the model captured the non-random patterns (good sign).  
* **Standard Errors & Coefficients:** Shows the estimated value and variability of the parameters.

## **Next Steps**

We've successfully formalized our stationarity test and fitted our first ARIMA model. In **Module 4**, we will focus on the most important aspect: **Forecasting and Validation**. We will learn how to split time series data correctly and evaluate our model's performance using metrics like RMSE.

# **Module 4: Forecasting and Validation**

Forecasting is the final goal of time series analysis, but an accurate forecast is only valuable if it has been rigorously validated. Unlike typical machine learning, time series validation requires specific techniques to maintain the chronological order of the data.

## **Setup: Data Initialization and Model Re-fit**

We will start by re-initializing our simulated data and importing necessary tools, including metrics from Scikit-learn.

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statsmodels.tsa.arima.model import ARIMA  
from sklearn.metrics import mean\_squared\_error, mean\_absolute\_error  
from math import sqrt

\# \--- Data Simulation \---  
\# Weekly data over 100 weeks (\~2 years)  
dates \= pd.date\_range(start='2020-01-01', periods=100, freq='W')  
np.random.seed(42)  
trend \= np.linspace(50, 150, 100\)  
seasonality \= 20 \* np.sin(np.linspace(0, 4 \* np.pi, 100))   
noise \= np.random.normal(0, 5, 100\)  
sales \= trend \+ seasonality \+ noise

data \= pd.DataFrame({'Sales': sales}, index=dates)  
data.index.name \= 'Date'

print("Total Data Length:", len(data))

## **4.1 Time Series Train-Test Split**

The most common mistake in time series is random splitting. Since observations are dependent on previous values, we **must** split the data chronologically. The test set must consist of the *most recent* observations.

We will use  data points for training (historical data) and the remaining  data points for testing (the future period we want to predict).

\# Define the split point  
split\_point \= 80  
train \= data\['Sales'\].iloc\[:split\_point\]  
test \= data\['Sales'\].iloc\[split\_point:\]

print(f"Train set length: {len(train)}")  
print(f"Test set length: {len(test)}")  
print("Test period starts at:", test.index\[0\])

\# Visualize the split  
plt.figure(figsize=(12, 6))  
plt.plot(train, label='Training Data', color='\#0077b6')  
plt.plot(test, label='Testing Data (Actual Future)', color='red')  
plt.title('Time Series Train-Test Split')  
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()

## **4.2 Generating Forecasts**

We must re-fit our chosen ARIMA model (we used  in Module 3\) using **only** the training data.

The model\_fit.predict() method requires two arguments to generate forecasts:

1. start: The index of the first point to predict (which is the beginning of the test set).  
2. end: The index of the last point to predict (which is the end of the test set).

\# 1\. Fit the ARIMA model on the training data only  
order \= (1, 1, 1\)  
model \= ARIMA(train, order=order)  
model\_fit \= model.fit()

print(f"ARIMA{order} fitted successfully on training data.")

\# 2\. Define the start and end dates for the forecast  
start\_date \= test.index\[0\]  
end\_date \= test.index\[-1\]

\# 3\. Generate the predictions  
\# The 'start' and 'end' parameters can take date strings or index integers.  
forecast \= model\_fit.predict(start=start\_date, end=end\_date, dynamic=False)

print("\\n--- Forecast Head \---")  
print(forecast.head())

### **Understanding dynamic=False**

* **dynamic=False (One-step-ahead prediction):** The model uses the *actual* value from the previous time step to predict the current time step. This is ideal for validation because it shows how well the model performs given the most recent actual data.  
* **dynamic=True (Multi-step prediction):** The model uses its *own previous forecast* as input for the next time step. This is how the model is used for true, future forecasting.

## **4.3 Evaluating Model Performance**

We use standard regression metrics to quantify the accuracy of our forecasts by comparing the forecast series against the test series.

### **Key Metrics:**

* **Mean Absolute Error (MAE):** The average magnitude of the errors. Easier to interpret as it is in the same units as the data.  
* **Root Mean Squared Error (RMSE):** The square root of the average squared errors. Penalizes larger errors more heavily.

\# Calculate evaluation metrics  
mae \= mean\_absolute\_error(test, forecast)  
rmse \= sqrt(mean\_squared\_error(test, forecast))

print(f"\\n--- Model Evaluation (ARIMA{order}) \---")  
print(f"Mean Absolute Error (MAE): {mae:.2f}")  
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

\# The RMSE value tells us the average magnitude of error in sales units.  
\# If our RMSE is 8.56, on average, our forecast is off by 8.56 sales units.

## **4.4 Visualizing Forecast vs. Actual**

Plotting the results gives us an intuitive understanding of where the model succeeded and where it failed.

plt.figure(figsize=(14, 7))

\# 1\. Plot the training data  
plt.plot(train, label='Training Data', color='gray', alpha=0.7)

\# 2\. Plot the actual test data  
plt.plot(test, label='Actual Test Data', color='red', linewidth=2)

\# 3\. Plot the forecast  
plt.plot(forecast, label='ARIMA Forecast', color='green', linestyle='--', linewidth=2)

plt.title(f'ARIMA{order} Forecast vs. Actual Sales Data')  
plt.xlabel('Date')  
plt.ylabel('Sales Volume')  
plt.legend()  
plt.grid(True, linestyle=':', alpha=0.6)  
plt.show()

## **Next Steps**

We've mastered the classical ARIMA approach and validated our model. However, ARIMA struggles with highly seasonal and complex data. In **Module 5**, we will tackle this by introducing more advanced models that handle both trend and seasonality implicitly: **SARIMA (Seasonal ARIMA)** and **Prophet**.

# **Module 5: Advanced Time Series Models and ML Approaches**

While ARIMA is a powerful foundational model, real-world data often exhibits strong and multiple seasonal patterns that require more sophisticated approaches. This module explores **SARIMA**, the user-friendly **Prophet** model, and the critical technique of **feature engineering** for traditional Machine Learning algorithms.

## **Setup: Data Initialization and Imports**

We continue with our sales data and introduce the required libraries for SARIMA and Prophet. Note that Prophet requires its own data format.

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statsmodels.tsa.statespace.sarimax import SARIMAX  
from sklearn.metrics import mean\_squared\_error  
from math import sqrt  
\# Install prophet if needed: \!pip install prophet  
from prophet import Prophet 

\# \--- Data Simulation \---  
\# Weekly data over 100 weeks (\~2 years)  
dates \= pd.date\_range(start='2020-01-01', periods=100, freq='W')  
np.random.seed(42)  
trend \= np.linspace(50, 150, 100\)  
\# A strong seasonal component is crucial here  
seasonality \= 30 \* np.sin(np.linspace(0, 8 \* np.pi, 100))   
noise \= np.random.normal(0, 7, 100\)  
sales \= trend \+ seasonality \+ noise

data \= pd.DataFrame({'Sales': sales}, index=dates)  
data.index.name \= 'Date'

\# Train/Test Split (as defined in Module 4\)  
split\_point \= 80  
train \= data.iloc\[:split\_point\]  
test \= data.iloc\[split\_point:\]

print(f"Data Head:\\n{data.head()}")

## **5.1 Seasonal ARIMA (SARIMAX)**

**SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) extends the ARIMA model by adding seasonal components.

The notation is: 

| Parameter | Meaning |
| :---- | :---- |
|  | Non-seasonal orders (same as ARIMA) |
|  | Seasonal orders (AR, Integration, MA) |
|  | The length of the seasonal cycle (e.g.,  for quarterly,  for monthly,  for weekly). |

For our weekly data, let's assume a quarterly cycle, so  (13 weeks/quarter). We will use a simple seasonal order .

\# SARIMA Model Order:  
\# Non-seasonal: (1, 1, 1\) \- from Module 3 analysis  
\# Seasonal: (1, 1, 1, 13\) \- assuming P=1, D=1 (seasonal differencing), Q=1, s=13 (quarterly cycle in weekly data)  
order \= (1, 1, 1\)  
seasonal\_order \= (1, 1, 1, 13\)

\# Fit SARIMAX on the training data  
sarima\_model \= SARIMAX(train\['Sales'\],   
                       order=order,   
                       seasonal\_order=seasonal\_order,   
                       enforce\_stationarity=False,   
                       enforce\_invertibility=False)

sarima\_fit \= sarima\_model.fit(disp=False)  
print("SARIMAX Model Fitted.")

\# Generate forecast  
start\_index \= len(train)  
end\_index \= len(data) \- 1

sarima\_forecast \= sarima\_fit.predict(start=start\_index, end=end\_index, dynamic=False)

\# Evaluate  
rmse\_sarima \= sqrt(mean\_squared\_error(test\['Sales'\], sarima\_forecast))  
print(f"SARIMA{order}x{seasonal\_order} RMSE: {rmse\_sarima:.2f}")

\# Plot  
plt.figure(figsize=(14, 6))  
plt.plot(train\['Sales'\], label='Training Data', color='gray')  
plt.plot(test\['Sales'\], label='Actual Test Data', color='red', linewidth=2)  
plt.plot(sarima\_forecast, label='SARIMA Forecast', color='blue', linestyle='--')  
plt.title('SARIMA Forecast vs. Actual')  
plt.legend()  
plt.grid(True, alpha=0.6)  
plt.show()

## **5.2 Facebook Prophet**

The **Prophet** library (developed by Meta, formerly Facebook) is optimized for business forecasts that often have strong yearly and weekly seasonality, and a clear linear or non-linear trend. It is very user-friendly and handles missing data automatically.

Prophet requires the input DataFrame to have two specific columns:

1. **ds** (datetime stamp)  
2. **y** (the value to be forecasted)

### **Example: Forecasting with Prophet**

\# 1\. Prepare data for Prophet  
prophet\_data \= data.reset\_index().rename(columns={'Date': 'ds', 'Sales': 'y'})  
prophet\_train \= prophet\_data.iloc\[:split\_point\]  
prophet\_test\_dates \= prophet\_data.iloc\[split\_point:\]

\# 2\. Initialize and Fit the model  
m \= Prophet(  
    yearly\_seasonality=True,   
    weekly\_seasonality=False, \# We have weekly data, so this may not be relevant  
    daily\_seasonality=False,  
    seasonality\_mode='multiplicative' \# Good for trend-dependent seasonality  
)  
m.fit(prophet\_train)

\# 3\. Create a DataFrame of future dates (periods to forecast)  
future\_dates \= m.make\_future\_dataframe(periods=len(test), freq='W', include\_history=False)

\# 4\. Generate the forecast  
prophet\_forecast \= m.predict(future\_dates)

\# 5\. Extract and Evaluate  
prophet\_preds \= prophet\_forecast\['yhat'\].values  
rmse\_prophet \= sqrt(mean\_squared\_error(prophet\_test\_dates\['y'\], prophet\_preds))  
print(f"\\nProphet RMSE: {rmse\_prophet:.2f}")

\# 6\. Plotting the forecast  
fig1 \= m.plot(prophet\_forecast)  
plt.title('Prophet Forecast')  
plt.show()

\# 7\. Plotting the components (trend, seasonality)  
fig2 \= m.plot\_components(prophet\_forecast)  
plt.show()

## **5.3 Time Series as a Supervised Learning Problem**

For complex data, you can convert the time series problem into a supervised learning problem by creating **lag features**. This allows you to use models like **Random Forest, Gradient Boosting (XGBoost)**, or standard **Linear Regression**.

### **Concept: Feature Engineering**

You predict  (Sales today) using:

*  (Sales yesterday)  
*  (Sales last week)  
* External variables (**Exogenous variables**, like price, promotions, or weather).  
* Date features (Day of week, Month, Quarter).

### **Example: Creating Lag and Date Features**

\# 1\. Create Lag Features  
data\['Lag\_1'\] \= data\['Sales'\].shift(1)  
data\['Lag\_7'\] \= data\['Sales'\].shift(7) \# Last week's sales

\# 2\. Create Date Features (Exogenous Variables)  
data\['Week\_of\_Year'\] \= data.index.isocalendar().week.astype(int)  
data\['Quarter'\] \= data.index.quarter

\# 3\. Clean up NaNs created by lagging  
data\_ml \= data.dropna()

print("\\nML Data Head (with features):")  
print(data\_ml.head())

\# 4\. Define Features (X) and Target (y)  
features \= \['Lag\_1', 'Lag\_7', 'Week\_of\_Year', 'Quarter'\]  
X \= data\_ml\[features\]  
y \= data\_ml\['Sales'\]

\# 5\. Split ML data chronologically  
ml\_split\_point \= len(data\_ml) \- 15 \# Use the last 15 points for testing  
X\_train, X\_test \= X.iloc\[:ml\_split\_point\], X.iloc\[ml\_split\_point:\]  
y\_train, y\_test \= y.iloc\[:ml\_split\_point\], y.iloc\[ml\_split\_point:\]

\# 6\. Fit a Simple Linear Regression Model  
from sklearn.linear\_model import LinearRegression  
ml\_model \= LinearRegression()  
ml\_model.fit(X\_train, y\_train)

\# 7\. Predict and Evaluate  
ml\_preds \= ml\_model.predict(X\_test)  
rmse\_ml \= sqrt(mean\_squared\_error(y\_test, ml\_preds))  
print(f"\\nMachine Learning (Linear Regression) RMSE: {rmse\_ml:.2f}")

\# Note: For multi-step forecasting with ML, you must use a 'recursive' approach,  
\# where you use the current prediction as the input lag for the next prediction.

## **Section Conclusion**

We've completed the Time Series Analysis Section. You can now:

1. **Pre-process** time series data (resample, decompose, difference).  
2. **Test for stationarity** (ADF test).  
3. **Identify parameters** () using ACF/PACF.  
4. **Fit and evaluate** classical ARIMA and SARIMA models.  
5. **Use advanced libraries** like Prophet for complex forecasts.  
6. **Apply Machine Learning** techniques by generating lag and external features.

The key takeaway is that the best model depends entirely on your data structure. Always start with EDA\!