You're right! Let's go through the derivation to see where the formula $\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$ actually comes from.

In simple linear regression, the goal is to find the best-fitting line for a set of data points $(X_i, Y_i)$ by minimizing the error between the actual values and the predicted values.

---

### Derivation of the Slope ($ \beta_1 $) Using the Least Squares Method

The line we’re trying to fit has the equation:
$$
Y = \beta_0 + \beta_1 X
$$
where:
- $ \beta_0 $ is the y-intercept,
- $ \beta_1 $ is the slope.

To find the optimal values of $ \beta_0 $ and $ \beta_1 $, we use the **least squares method** to minimize the sum of squared residuals (the squared differences between the actual and predicted $ Y $-values).

#### Step 1: Define the Error (Residual) Function

The residual for each point $(X_i, Y_i)$ is:
$$
e_i = Y_i - (\beta_0 + \beta_1 X_i)
$$
The goal is to minimize the **sum of squared residuals**:
$$
S = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} \left(Y_i - \beta_0 - \beta_1 X_i\right)^2
$$

#### Step 2: Minimize $ S $ with Respect to $ \beta_0 $ and $ \beta_1 $

To find the values of $ \beta_0 $ and $ \beta_1 $ that minimize $ S $, we take partial derivatives of $ S $ with respect to $ \beta_0 $ and $ \beta_1 $ and set them equal to zero.

1. **Derivative with respect to $ \beta_0 $:**
   $$
   \frac{\partial S}{\partial \beta_0} = -2 \sum_{i=1}^{n} \left(Y_i - \beta_0 - \beta_1 X_i\right) = 0
   $$
   Simplifying, we get:
   $$
   \sum_{i=1}^{n} Y_i = n \beta_0 + \beta_1 \sum_{i=1}^{n} X_i
   $$
   Dividing by $ n $, we find:
   $$
   \bar{Y} = \beta_0 + \beta_1 \bar{X}
   $$
   This tells us that the intercept $ \beta_0 $ can be expressed as:
   $$
   \beta_0 = \bar{Y} - \beta_1 \bar{X}
   $$

2. **Derivative with respect to $ \beta_1 $:**
   $$
   \frac{\partial S}{\partial \beta_1} = -2 \sum_{i=1}^{n} X_i \left(Y_i - \beta_0 - \beta_1 X_i\right) = 0
   $$
   Substituting $ \beta_0 = \bar{Y} - \beta_1 \bar{X} $, we can rewrite this as:
   $$
   \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y}) = \beta_1 \sum_{i=1}^{n} (X_i - \bar{X})^2
   $$

   Solving for $ \beta_1 $, we get:
   $$
   \beta_1 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
   $$

#### Step 3: Recognize Covariance and Variance Notation

The expression for $ \beta_1 $ can now be rewritten in terms of **covariance** and **variance**:
- The **covariance** between $ X $ and $ Y $, $ \text{Cov}(X, Y) $, is defined as:
  $$
  \text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
  $$
- The **variance** of $ X $, $ \text{Var}(X) $, is defined as:
  $$
  \text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2
  $$

Thus, we can express the slope $ \beta_1 $ as:
$$
\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

---

### Conclusion

The formula $ \beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} $ is derived from minimizing the sum of squared residuals using the least squares approach. This form shows that the slope $ \beta_1 $ represents the rate at which $ Y $ changes with respect to $ X $, based on the relationship between the variance of $ X $ and the covariance of $ X $ and $ Y $.