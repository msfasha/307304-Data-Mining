In matrix form, the derivation for the slope of simple linear regression can be elegantly represented using vector and matrix operations. Here’s how we approach it.

Let’s define the **simple linear regression model** in a general form for multiple observations:
$$
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
$$
where:
- $ \mathbf{Y} $ is an $ n \times 1 $ vector of observed values (the dependent variable),
- $ \mathbf{X} $ is an $ n \times 2 $ matrix of predictors (the independent variable),
- $ \boldsymbol{\beta} $ is a $ 2 \times 1 $ vector of parameters (including intercept and slope),
- $ \boldsymbol{\epsilon} $ is an $ n \times 1 $ vector of errors.

For simple linear regression with a single predictor, our design matrix $ \mathbf{X} $ is structured as follows:
$$
\mathbf{X} = \begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix}
$$
where:
- The first column (all 1’s) allows us to model the intercept $ \beta_0 $,
- The second column contains values of $ X $, the independent variable.

The vector $ \boldsymbol{\beta} $ is:
$$
\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}
$$

### Step 1: Express the Least Squares Objective in Matrix Form

The least squares method aims to minimize the sum of squared residuals, which can be written in matrix form as:
$$
S = (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})
$$
Expanding $ S $ yields:
$$
S = \mathbf{Y}^T \mathbf{Y} - 2 \mathbf{Y}^T \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$

### Step 2: Take Derivative with Respect to $ \boldsymbol{\beta} $ and Set to Zero

To find the minimum, we differentiate $ S $ with respect to $ \boldsymbol{\beta} $ and set it equal to zero:
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = -2 \mathbf{X}^T \mathbf{Y} + 2 \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = 0
$$
Simplifying, we obtain the **normal equation**:
$$
\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{Y}
$$

### Step 3: Solve for $ \boldsymbol{\beta} $

Assuming $ \mathbf{X}^T \mathbf{X} $ is invertible, we can solve for $ \boldsymbol{\beta} $ as:
$$
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

### Interpreting the Solution

In this solution:
- The intercept $ \beta_0 $ and slope $ \beta_1 $ are estimated together in $ \boldsymbol{\beta} $.
- The term $ (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} $ gives us the values that minimize the squared differences between the observed and predicted $ Y $-values, providing the least squares estimates.

---

This matrix form derivation is more generalized and compact, allowing the same principles to be extended to multiple linear regression with more predictors. It provides an efficient way to solve for the regression coefficients, particularly when there are multiple predictors.