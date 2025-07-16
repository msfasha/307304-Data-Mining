Using **Maximum Likelihood Estimation (MLE)** to derive the parameters of linear regression involves assuming a probabilistic model for the data and finding the parameter values that maximize the likelihood of the observed data. Here’s how the process works in detail.

---

### Linear Regression Model with Gaussian Assumptions

In simple linear regression, we have:
$$
Y = \beta_0 + \beta_1 X + \epsilon
$$
where $ Y $ is the dependent variable, $ X $ is the independent variable, $ \beta_0 $ is the intercept, $ \beta_1 $ is the slope, and $ \epsilon $ is an error term.

To apply MLE, we make the following assumptions:
1. **Linearity**: $ Y $ can be modeled as a linear function of $ X $ plus some noise.
2. **Gaussian Errors**: The errors $ \epsilon $ are independently and identically distributed (i.i.d.) and follow a normal distribution with mean zero and variance $ \sigma^2 $:
   $$
   \epsilon \sim \mathcal{N}(0, \sigma^2)
   $$
3. **Fixed $X$**: The independent variable $ X $ is fixed (non-random).

Under these assumptions, each observed $ Y_i $ given $ X_i $ follows a normal distribution:
$$
Y_i \mid X_i \sim \mathcal{N}(\beta_0 + \beta_1 X_i, \sigma^2)
$$

### Step 1: Likelihood Function

The probability density function of $ Y_i $ given $ X_i $ is:
$$
p(Y_i \mid X_i, \beta_0, \beta_1, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(Y_i - \beta_0 - \beta_1 X_i)^2}{2 \sigma^2} \right)
$$

Since we have $ n $ independent observations, the **likelihood function** for the entire dataset is the product of the probabilities for each observation:
$$
L(\beta_0, \beta_1, \sigma^2) = \prod_{i=1}^{n} p(Y_i \mid X_i, \beta_0, \beta_1, \sigma^2)
$$
Substitute the expression for $ p(Y_i \mid X_i, \beta_0, \beta_1, \sigma^2) $:
$$
L(\beta_0, \beta_1, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(Y_i - \beta_0 - \beta_1 X_i)^2}{2 \sigma^2} \right)
$$

### Step 2: Log-Likelihood Function

The **log-likelihood function** is often used instead of the likelihood for easier differentiation. Taking the natural log of $ L(\beta_0, \beta_1, \sigma^2) $, we get:
$$
\ln L(\beta_0, \beta_1, \sigma^2) = \sum_{i=1}^{n} \ln \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(Y_i - \beta_0 - \beta_1 X_i)^2}{2 \sigma^2} \right) \right)
$$

Expanding this, we have:
$$
\ln L(\beta_0, \beta_1, \sigma^2) = -\frac{n}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

### Step 3: Maximizing the Log-Likelihood

To find the maximum likelihood estimates of $ \beta_0 $ and $ \beta_1 $, we take partial derivatives of $ \ln L $ with respect to $ \beta_0 $, $ \beta_1 $, and $ \sigma^2 $, and set them equal to zero.

1. **Derivative with respect to $ \beta_0 $**:
   $$
   \frac{\partial \ln L}{\partial \beta_0} = \frac{1}{\sigma^2} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i) = 0
   $$

2. **Derivative with respect to $ \beta_1 $**:
   $$
   \frac{\partial \ln L}{\partial \beta_1} = \frac{1}{\sigma^2} \sum_{i=1}^{n} X_i (Y_i - \beta_0 - \beta_1 X_i) = 0
   $$

These equations are equivalent to the **normal equations** in linear regression:
$$
\sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i) = 0
$$
$$
\sum_{i=1}^{n} X_i (Y_i - \beta_0 - \beta_1 X_i) = 0
$$

Solving these equations yields:
$$
\beta_1 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
$$
$$
\beta_0 = \bar{Y} - \beta_1 \bar{X}
$$

These are the same estimates obtained using the ordinary least squares (OLS) method.

3. **Derivative with respect to $ \sigma^2 $**:
   $$
   \frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2 \sigma^2} + \frac{1}{2 \sigma^4} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2 = 0
   $$

Solving for $ \sigma^2 $, we get:
$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

This is the maximum likelihood estimate of the variance, equivalent to the sample variance of the residuals.

---

### Summary

Using MLE, we arrive at the same parameter estimates for $ \beta_0 $ and $ \beta_1 $ as with the least squares approach, as well as an estimate for $ \sigma^2 $, the variance of the errors. The MLE approach provides a probabilistic basis for estimating the parameters in linear regression, useful for inference and hypothesis testing.