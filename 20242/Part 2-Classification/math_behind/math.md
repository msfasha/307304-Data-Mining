# **Logistic Regression: Mathematical Background**

## 1. **Introduction**
Logistic Regression is a statistical model used for **classification tasks**. It predicts the probability of a binary or multinomial outcome. The foundation of logistic regression is the **sigmoid function**, which transforms a linear equation into a probabilistic output.

---

## 2. **The Sigmoid Function**
The sigmoid (or logistic) function is defined as:

$
\sigma(z) = \frac{1}{1 + e^{-z}}
$

### Properties of the Sigmoid:
1. Outputs values between $0$ and $1$, making it suitable for probability estimation.
2. The function is **S-shaped** (hence the term "sigmoid"), with:
   - $ \sigma(z) \to 0 $ as $ z \to -\infty $
   - $ \sigma(z) \to 1 $ as $ z \to +\infty $
3. The derivative is:
   $
   \sigma'(z) = \sigma(z) (1 - \sigma(z))
   $
   This property is useful for optimization (e.g., Gradient Descent).

### Why Sigmoid for Classification?
In binary classification, we need to predict probabilities. A linear model ($ z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p $) can output values from $-\infty$ to $+\infty$, which is not interpretable as probabilities. The sigmoid maps this range into $ (0, 1) $.

---

## 3. **Binary Logistic Regression**
In binary logistic regression, we predict the probability of an outcome $ y \in \{0, 1\} $. The model is:

$
p(y = 1 | X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}
$

### Odds and Log-Odds (Logit)
The odds of an event are:
$
\text{Odds} = \frac{p}{1-p}
$

The log-odds (logit function) is the natural logarithm of the odds:
$
\log\left(\frac{p}{1-p}\right) = z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$

### Decision Rule
To classify:
- If $ p \geq 0.5 $, predict $ y = 1 $.
- Otherwise, predict $ y = 0 $.

---

## 4. **Loss Function for Binary Logistic Regression**
The **likelihood function** measures how well the model predicts the observed data. For binary outcomes:

$
L(\beta) = \prod_{i=1}^{n} \left[\sigma(z_i)\right]^{y_i} \left[1 - \sigma(z_i)\right]^{1-y_i}
$

Taking the log of the likelihood (log-likelihood):
$
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]
$

The **negative log-likelihood** (log-loss) is minimized during training:
$
\mathcal{L}(\beta) = -\ell(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]
$

### Intuition Behind the Loss Function:
1. If $ y_i = 1 $, the first term $ \log(\sigma(z_i)) $ ensures that high probabilities for $ y_i = 1 $ are rewarded.
2. If $ y_i = 0 $, the second term $ \log(1 - \sigma(z_i)) $ ensures that low probabilities for $ y_i = 1 $ are penalized.

---

## 5. **Multinomial Logistic Regression**
For multinomial classification, where $ y \in \{1, 2, \dots, K\} $, logistic regression generalizes to **Softmax Regression**.

### Softmax Function
The Softmax function outputs probabilities for $ K $ classes:
$
p(y = k | X) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$
where $ z_k = \beta_{k0} + \beta_{k1}x_1 + \dots + \beta_{kp}x_p $.

### Decision Rule
Classify $ X $ into the class with the highest probability:
$
\hat{y} = \arg\max_k p(y = k | X)
$

---

### Loss Function for Multinomial Logistic Regression
The likelihood function for $ n $ samples is:
$
L(\beta) = \prod_{i=1}^{n} \prod_{k=1}^{K} \left[p(y_i = k | X_i)\right]^{\mathbb{1}(y_i = k)}
$
where $ \mathbb{1}(y_i = k) $ is an indicator function that equals 1 if $ y_i = k $, and 0 otherwise.

The log-likelihood becomes:
$
\ell(\beta) = \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log\left(p(y_i = k | X_i)\right)
$

The negative log-likelihood (cross-entropy loss):
$
\mathcal{L}(\beta) = - \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log\left(p(y_i = k | X_i)\right)
$

---

### Comparing Binary and Multinomial Loss:
- **Binary Logistic Regression**: Special case of multinomial regression with $ K = 2 $.
- **Multinomial Logistic Regression**: Handles $ K > 2 $ classes and uses Softmax to compute probabilities.

---

## 6. **Optimization**
The loss function for both binary and multinomial logistic regression is non-linear, requiring **iterative optimization algorithms** like:
1. Gradient Descent
2. Stochastic Gradient Descent (SGD)
3. Quasi-Newton Methods (e.g., L-BFGS)

The gradients are computed using:
- For binary: $\nabla \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \beta}$
- For multinomial: Gradients are derived similarly but involve all $ K $ classes.

---

## 7. **Key Takeaways**
1. Logistic regression transforms a linear model into probabilistic outputs using the sigmoid (binary) or softmax (multinomial) function.
2. The log-loss is minimized during training, ensuring accurate probability predictions.
3. For optimization, iterative methods like L-BFGS are used, as no closed-form solution exists.

This framework connects theory to practice, emphasizing the mathematical foundation behind logistic regression models.