# Mathematical Exercises for Linear Regression and Classification 🧮

This document contains theoretical and mathematical exercises specifically designed for the linear regression and classification modules of the Machine Learning curriculum. These exercises complement the practical programming exercises in `EXERCISES.md` with mathematical derivations, theoretical analysis, and numerical applications.

---

## Linear Regression Exercises 📈

### Exercise LR-1: Gradient Descent for Linear Regression
**(5 marks)**

Consider a simple linear regression model with one feature: y = w₁x + w₀, where w = [w₀, w₁]ᵀ are the parameters. Given the loss function:

J(w) = (1/2N) Σᵢ₌₁ᴺ (wᵀxᵢ - tᵢ)²

where xᵢ = [1, xᵢ]ᵀ (augmented with bias term) and tᵢ is the target value.

1. **Derive the gradient** ∇J(w) with respect to w. Show all steps. *(2 marks)*

2. **Write the gradient descent update rule** for both w₀ and w₁ with learning rate η. *(1 mark)*

3. **Apply the formula** to the following data with initial parameters w⁽⁰⁾ = [0.5, 1.0]ᵀ and learning rate η = 0.1:
   - x₁ = 2, t₁ = 5
   - x₂ = 3, t₂ = 7
   
   Calculate w⁽¹⁾ after one iteration of batch gradient descent. *(2 marks)*

---

### Exercise LR-2: Ridge Regression Analysis
**(6 marks)**

Consider the ridge regression loss function:

J(w) = (1/2N) Σᵢ₌₁ᴺ (wᵀxᵢ - tᵢ)² + (λ/2)||w||²

where λ > 0 is the regularization parameter.

1. **Derive the gradient** ∇J(w) including the regularization term. *(2 marks)*

2. **Find the closed-form solution** for the optimal weights w* by setting ∇J(w*) = 0. Express your answer in matrix form. *(2 marks)*

3. **Numerical application**: Given the design matrix X and target vector t:
   ```
   X = [1  2]    t = [3]
       [1  4]        [5]
       [1  6]        [8]
   ```
   Calculate w* for λ = 0.5. Show all matrix computations. *(2 marks)*

---

### Exercise LR-3: Convergence Analysis
**(4 marks)**

For the gradient descent algorithm applied to linear regression:

wᵏ⁺¹ = wᵏ - η∇J(wᵏ)

where J(w) = (1/2N)||Xw - t||².

1. **Express the Hessian matrix** H = ∇²J(w) in terms of X. *(1 mark)*

2. **State the condition** on the learning rate η for guaranteed convergence in terms of the eigenvalues of H. *(1 mark)*

3. **For the matrix X from Exercise LR-2**, calculate the maximum eigenvalue of XᵀX and determine the maximum learning rate that guarantees convergence. *(2 marks)*

---

## Linear Classification Exercises 🎯

### Exercise LC-1: Logistic Regression Derivation
**(6 marks)**

Consider binary logistic regression with the sigmoid function σ(z) = 1/(1 + e⁻ᶻ) and log-likelihood:

L(w) = Σᵢ₌₁ᴺ [tᵢ log σ(wᵀxᵢ) + (1-tᵢ) log(1-σ(wᵀxᵢ))]

where tᵢ ∈ {0,1} are the binary labels.

1. **Show that** dσ(z)/dz = σ(z)(1-σ(z)). *(1 mark)*

2. **Derive the gradient** ∇L(w) with respect to w. Show all steps using the chain rule. *(3 marks)*

3. **Write the gradient ascent update rule** (since we maximize likelihood) for w with learning rate η. *(1 mark)*

4. **Apply to the data point** x = [1, 2, -1]ᵀ, t = 1, with current weights w = [0.1, 0.5, -0.2]ᵀ and η = 0.3. Calculate the weight update. *(1 mark)*

---

### Exercise LC-2: Multiclass Classification
**(5 marks)**

For multiclass logistic regression with K classes, the softmax function is:

p(y = k|x, W) = exp(wₖᵀx) / Σⱼ₌₁ᴷ exp(wⱼᵀx)

where W = [w₁, w₂, ..., wₖ] is the weight matrix.

1. **Show that** Σₖ₌₁ᴷ p(y = k|x, W) = 1. *(1 mark)*

2. **Derive the gradient** of the negative log-likelihood with respect to wₖ for a single training example (x, t), where t is a one-hot encoded vector. *(3 marks)*

3. **For K = 3 classes**, given:
   - x = [1, 2]ᵀ
   - W = [0.1  0.3; 0.2  0.1; -0.1  0.2]
   - True class: t = [0, 1, 0]ᵀ
   
   Calculate the probability distribution and the gradient update for w₂. *(1 mark)*

---

### Exercise LC-3: Perceptron Algorithm Analysis
**(4 marks)**

The perceptron algorithm updates weights according to:

w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ + η(tᵢ - ŷᵢ)xᵢ

where ŷᵢ = sign(w⁽ᵗ⁾ᵀxᵢ) and tᵢ ∈ {-1, +1}.

1. **Explain why** the perceptron only updates weights when there is a misclassification. *(1 mark)*

2. **Given the linearly separable dataset**:
   ```
   x₁ = [1, 1]ᵀ,  t₁ = +1
   x₂ = [1, 2]ᵀ,  t₂ = +1  
   x₃ = [2, 1]ᵀ,  t₃ = -1
   x₄ = [2, 2]ᵀ,  t₄ = -1
   ```
   
   Starting with w⁽⁰⁾ = [0, 0]ᵀ and η = 1, perform 3 iterations through the dataset and show the weight updates. *(2 marks)*

3. **State the perceptron convergence theorem** and explain what it guarantees about the algorithm's behavior on linearly separable data. *(1 mark)*

---

### Exercise LC-4: Decision Boundaries and Regularization
**(6 marks)**

Consider logistic regression with L2 regularization:

J(w) = -Σᵢ₌₁ᴺ [tᵢ log σ(wᵀxᵢ) + (1-tᵢ) log(1-σ(wᵀxᵢ))] + (λ/2)||w||²

1. **Derive the regularized gradient** ∇J(w). *(2 marks)*

2. **For 2D input (x₁, x₂) with bias**, write the equation of the decision boundary where p(y=1|x) = 0.5. *(1 mark)*

3. **Given the regularized logistic regression model** with weights w = [w₀, w₁, w₂]ᵀ = [1, -2, 3]ᵀ:
   - Find the decision boundary equation
   - Classify the points: A = [1, 1]ᵀ, B = [2, 0]ᵀ
   - Calculate the predicted probabilities for both points *(3 marks)*

---

## Advanced Integration Exercises 🔗

### Exercise ADV-1: Comparing Regression and Classification
**(5 marks)**

Consider the same dataset used for both regression and classification tasks:

```
x₁ = 1, y₁ = 0.8    (regression target) / t₁ = 1 (classification label)
x₂ = 2, y₂ = 0.3    (regression target) / t₂ = 0 (classification label)  
x₃ = 3, y₃ = 0.1    (regression target) / t₃ = 0 (classification label)
```

1. **Fit a linear regression** model y = w₁x + w₀ using least squares. Calculate w₀ and w₁. *(2 marks)*

2. **Fit a logistic regression** model using the same x values but binary labels t. Set up the likelihood equations (you don't need to solve numerically). *(2 marks)*

3. **Compare the decision-making process**: At what value of x would the linear regression predict y = 0.5? How does this compare to the logistic regression decision boundary? *(1 mark)*

---

### Exercise ADV-2: Optimization Comparison
**(4 marks)**

Compare gradient descent behavior for linear regression vs. logistic regression:

1. **Explain why** the linear regression cost function J(w) = (1/2N)||Xw - t||² is convex, while discussing the convexity of the logistic regression cost function. *(2 marks)*

2. **For identical datasets and identical initial weights**, would you expect gradient descent to converge faster for linear or logistic regression? Justify your answer by comparing the nature of their gradients. *(2 marks)*

---

## Solutions Guide 📝

### Tips for Solving These Exercises:

1. **Always start with the basic definitions** - understand what each symbol represents
2. **Use the chain rule systematically** - break down complex derivatives into manageable parts  
3. **Check your dimensions** - ensure matrix multiplications are valid
4. **Verify special cases** - test your formulas with simple examples
5. **Connect theory to practice** - relate mathematical results to algorithm behavior

### Key Formulas to Remember:

- **Linear Regression Gradient**: ∇J(w) = (1/N)Xᵀ(Xw - t)
- **Logistic Regression Gradient**: ∇L(w) = Xᵀ(t - σ(Xw))
- **Ridge Regularization**: Add λw to the gradient
- **Sigmoid Derivative**: σ'(z) = σ(z)(1 - σ(z))

---

## Sample Solution: Exercise LR-1
*(Provided as an example of expected solution format)*

**Exercise LR-1: Gradient Descent for Linear Regression**

**Part 1: Derive the gradient ∇J(w)**

Given: J(w) = (1/2N) Σᵢ₌₁ᴺ (wᵀxᵢ - tᵢ)²

Step 1: Expand the squared term
J(w) = (1/2N) Σᵢ₌₁ᴺ [(wᵀxᵢ)² - 2wᵀxᵢtᵢ + tᵢ²]

Step 2: Take partial derivative with respect to w
∂J/∂w = (1/2N) Σᵢ₌₁ᴺ [2wᵀxᵢ · xᵢ - 2tᵢxᵢ]
      = (1/N) Σᵢ₌₁ᴺ [wᵀxᵢxᵢ - tᵢxᵢ]
      = (1/N) Σᵢ₌₁ᴺ xᵢ(wᵀxᵢ - tᵢ)

Therefore: **∇J(w) = (1/N) Σᵢ₌₁ᴺ xᵢ(wᵀxᵢ - tᵢ)**

**Part 2: Gradient descent update rule**

**w⁽ᵏ⁺¹⁾ = w⁽ᵏ⁾ - η∇J(w⁽ᵏ⁾)**

For individual components:
- **w₀⁽ᵏ⁺¹⁾ = w₀⁽ᵏ⁾ - η(1/N) Σᵢ₌₁ᴺ (w⁽ᵏ⁾ᵀxᵢ - tᵢ)**
- **w₁⁽ᵏ⁺¹⁾ = w₁⁽ᵏ⁾ - η(1/N) Σᵢ₌₁ᴺ (w⁽ᵏ⁾ᵀxᵢ - tᵢ)xᵢ**

**Part 3: Numerical application**

Given: w⁽⁰⁾ = [0.5, 1.0]ᵀ, η = 0.1, N = 2
- Point 1: x₁ = [1, 2]ᵀ, t₁ = 5
- Point 2: x₂ = [1, 3]ᵀ, t₂ = 7

Step 1: Calculate predictions
- ŷ₁ = w⁽⁰⁾ᵀx₁ = 0.5(1) + 1.0(2) = 2.5
- ŷ₂ = w⁽⁰⁾ᵀx₂ = 0.5(1) + 1.0(3) = 3.5

Step 2: Calculate errors
- e₁ = ŷ₁ - t₁ = 2.5 - 5 = -2.5
- e₂ = ŷ₂ - t₂ = 3.5 - 7 = -3.5

Step 3: Calculate gradient
∇J(w⁽⁰⁾) = (1/2)[x₁e₁ + x₂e₂]
         = (1/2)[[1, 2]ᵀ(-2.5) + [1, 3]ᵀ(-3.5)]
         = (1/2)[[-2.5, -5.0]ᵀ + [-3.5, -10.5]ᵀ]
         = (1/2)[-6.0, -15.5]ᵀ
         = [-3.0, -7.75]ᵀ

Step 4: Update weights
w⁽¹⁾ = w⁽⁰⁾ - η∇J(w⁽⁰⁾)
     = [0.5, 1.0]ᵀ - 0.1[-3.0, -7.75]ᵀ
     = [0.5, 1.0]ᵀ + [0.3, 0.775]ᵀ
     = **[0.8, 1.775]ᵀ**

---

*These exercises are designed to deepen your mathematical understanding of linear models in machine learning. Work through them systematically, and don't hesitate to review the theoretical foundations in the course PDFs.*