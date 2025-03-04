# Solving Linear Reqression using Gradient Descent
### Setting Up the Problem

In simple linear regression, we want to find the best-fitting line:
$$
Y = \beta_0 + \beta_1 X
$$
for a set of points $(X_i, Y_i)$ where $i = 1, 2, \dots, n$.

Our goal is to minimize the **Mean Squared Error (MSE)** between the predicted and actual values. This is captured by the cost function:
$$
J(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - (\beta_0 + \beta_1 X_i) \right)^2
$$

### Step 1: Expand the Cost Function

To find the optimal values for $ \beta_0 $ and $ \beta_1 $, we need to minimize $ J(\beta_0, \beta_1) $. This requires us to calculate the derivatives of $ J $ with respect to $ \beta_0 $ and $ \beta_1 $, which will show us how changes in $ \beta_0 $ and $ \beta_1 $ affect $ J $.

Let’s expand the squared term:
$$
J(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \beta_0 - \beta_1 X_i \right)^2
$$

### Step 2: Compute the Partial Derivative with Respect to $ \beta_0 $

The partial derivative of $ J $ with respect to $ \beta_0 $ tells us how $ J $ changes as $ \beta_0 $ changes, while keeping $ \beta_1 $ constant. 

1. Start by applying the chain rule:
   $$
   \frac{\partial J}{\partial \beta_0} = \frac{\partial}{\partial \beta_0} \left( \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \beta_0 - \beta_1 X_i \right)^2 \right)
   $$

2. Differentiate the sum term by term:
   $$
   \frac{\partial J}{\partial \beta_0} = \frac{1}{n} \sum_{i=1}^{n} 2 \left( Y_i - \beta_0 - \beta_1 X_i \right) \cdot (-1)
   $$

3. Simplify by pulling out constants:
   $$
   \frac{\partial J}{\partial \beta_0} = -\frac{2}{n} \sum_{i=1}^{n} \left( Y_i - \beta_0 - \beta_1 X_i \right)
   $$

### Step 3: Compute the Partial Derivative with Respect to $ \beta_1 $

Next, we compute the partial derivative of $ J $ with respect to $ \beta_1 $ to understand how $ J $ changes as $ \beta_1 $ changes, while keeping $ \beta_0 $ constant.

1. Again, start with the chain rule:
   $$
   \frac{\partial J}{\partial \beta_1} = \frac{\partial}{\partial \beta_1} \left( \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \beta_0 - \beta_1 X_i \right)^2 \right)
   $$

2. Differentiate the sum term by term:
   $$
   \frac{\partial J}{\partial \beta_1} = \frac{1}{n} \sum_{i=1}^{n} 2 \left( Y_i - \beta_0 - \beta_1 X_i \right) \cdot (-X_i)
   $$

3. Simplify by pulling out constants:
   $$
   \frac{\partial J}{\partial \beta_1} = -\frac{2}{n} \sum_{i=1}^{n} X_i \left( Y_i - \beta_0 - \beta_1 X_i \right)
   $$

### Step 4: Gradient Descent Update Rules

The gradients we derived give us the direction in which $ J $ increases or decreases. In gradient descent, we move in the opposite direction of the gradient to minimize $ J $.

The update rules for $ \beta_0 $ and $ \beta_1 $ with learning rate $ \alpha $ are:
$$
\beta_0 := \beta_0 - \alpha \cdot \frac{\partial J}{\partial \beta_0}
$$
$$
\beta_1 := \beta_1 - \alpha \cdot \frac{\partial J}{\partial \beta_1}
$$

Substituting the gradients we derived:

$$
\beta_0 := \beta_0 + \alpha \cdot \frac{2}{n} \sum_{i=1}^{n} \left( Y_i - \beta_0 - \beta_1 X_i \right)
$$
$$
\beta_1 := \beta_1 + \alpha \cdot \frac{2}{n} \sum_{i=1}^{n} X_i \left( Y_i - \beta_0 - \beta_1 X_i \right)
$$

---

### Summary

Using gradient descent, we iteratively update $ \beta_0 $ and $ \beta_1 $ based on the gradients derived above. Each step moves us closer to the values of $ \beta_0 $ and $ \beta_1 $ that minimize the cost function $ J $, ultimately fitting the best line to the data.