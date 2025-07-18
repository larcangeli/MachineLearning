# University Machine Learning Exam Exercises 📝

This document contains university-level machine learning exercises designed for exam preparation. Each exercise combines mathematical theory with practical implementation, similar to what you would encounter in a formal ML course examination.

**Instructions:**
- Show all mathematical derivations clearly
- Implement code solutions where specified
- Explain theoretical reasoning for all answers
- Time allocation: 30-45 minutes per exercise

---

## Exercise 1: Linear Regression and Gradient Descent Foundations
**(4 marks)**

Consider a simple linear regression problem with one feature. Given the loss function:
```
J(w₀, w₁) = (1/2N) Σᵢ₌₁ᴺ (w₀ + w₁xᵢ - yᵢ)²
```

**Part 1 (1.5 marks):** Derive the gradient descent update rules for both parameters w₀ and w₁.

**Part 2 (1.5 marks):** Given the dataset:
- x = [1, 2, 3], y = [2, 4, 5]
- Initial parameters: w₀⁽⁰⁾ = 0, w₁⁽⁰⁾ = 0  
- Learning rate: α = 0.1

Calculate the parameter values after one gradient descent iteration.

**Part 3 (1 mark):** Implement the gradient descent algorithm in Python and verify your manual calculation. What happens to the convergence if you increase the learning rate to α = 2.0?

---

## Exercise 2: Logistic Regression and Maximum Likelihood
**(5 marks)**

For binary classification, consider the logistic regression model:
```
P(y=1|x) = σ(w^T x) = 1/(1 + e^(-w^T x))
```

**Part 1 (2 marks):** Derive the log-likelihood function for a dataset {(xᵢ, yᵢ)}ᵢ₌₁ᴺ where yᵢ ∈ {0,1}. Then derive the gradient of the negative log-likelihood with respect to the weight vector w.

**Part 2 (1.5 marks):** Given the training point x₁ = [1, -1, 2] with label y₁ = 1, and current weights w = [0.5, -0.3, 0.2], calculate:
- The predicted probability P(y=1|x₁)
- The gradient contribution from this single point
- The weight update using learning rate α = 0.5

**Part 3 (1.5 marks):** Implement logistic regression from scratch in Python. Compare the decision boundary with sklearn's implementation on a 2D dataset. Explain why logistic regression produces a linear decision boundary.

---

## Exercise 3: Bias-Variance Decomposition
**(4 marks)**

Consider a regression problem where the true relationship is y = f(x) + ε, with ε ~ N(0, σ²).

**Part 1 (2 marks):** For a learning algorithm that produces predictor ĥ(x), derive the bias-variance decomposition:
```
E[(y - ĥ(x))²] = Bias²[ĥ(x)] + Var[ĥ(x)] + σ²
```
Define each component clearly.

**Part 2 (1 mark):** Consider two models:
- Model A: Always predicts ĥₐ(x) = c (constant)
- Model B: k-nearest neighbors with k=1

For each model, qualitatively describe whether it has high/low bias and high/low variance. Justify your answers.

**Part 3 (1 mark):** Implement a simulation in Python where you generate multiple training sets from f(x) = x² + noise, train both polynomial regression (degree 1 and degree 5) on each, and empirically estimate bias and variance for a test point x₀ = 0.5.

---

## Exercise 4: Cross-Validation and Model Selection
**(4 marks)**

You are tasked with selecting the optimal polynomial degree for regression using k-fold cross-validation.

**Part 1 (1.5 marks):** Explain the difference between:
- Training error vs. validation error vs. test error
- k-fold CV vs. leave-one-out CV vs. holdout validation
- Model selection vs. model assessment

**Part 2 (1.5 marks):** Given a dataset with N=12 samples and 3-fold CV, list all possible training/validation splits. If you observe the following validation MSE values for different polynomial degrees:
- Degree 1: [2.3, 2.1, 2.4]
- Degree 2: [1.8, 1.7, 1.9] 
- Degree 3: [2.1, 2.0, 2.3]

Which degree would you select and why? Calculate the mean and standard error for each.

**Part 3 (1 mark):** Implement stratified k-fold cross-validation from scratch for classification. Demonstrate on the Iris dataset that class proportions are preserved in each fold.

---

## Exercise 5: PAC Learning and Sample Complexity
**(5 marks)**

Consider the concept class of axis-aligned rectangles in ℝ².

**Part 1 (2 marks):** Calculate the VC dimension of axis-aligned rectangles. Provide:
- A set of 4 points that can be shattered
- An explanation of why no set of 5 points can be shattered
- The formal VC dimension value

**Part 2 (1.5 marks):** Using the PAC learning framework, if we want to learn a concept with error at most ε = 0.1 with confidence at least δ = 0.05, what is the minimum sample complexity required? Use the VC bound:
```
m ≥ (1/ε)[ln(|H|) + ln(1/δ)]
```
where |H| is related to the VC dimension.

**Part 3 (1.5 marks):** Implement a simple rectangle learner in Python that finds the smallest axis-aligned rectangle containing all positive examples. Test it on synthetic 2D data and verify that it satisfies the consistency requirement of PAC learning.

---

## Exercise 6: Kernel Methods and the Kernel Trick
**(4 marks)**

Consider the polynomial kernel k(x, x') = (x^T x' + 1)ᵈ.

**Part 1 (1.5 marks):** For d=2 and 2D input vectors, explicitly show the feature mapping φ(x) such that k(x, x') = φ(x)^T φ(x'). Calculate the dimensionality of the feature space.

**Part 2 (1 mark):** Given two vectors x₁ = [1, 2] and x₂ = [3, -1], compute:
- k(x₁, x₂) using the kernel function directly
- φ(x₁)^T φ(x₂) using your explicit feature mapping
- Verify they are equal

**Part 3 (1.5 marks):** Implement kernel PCA using the RBF kernel k(x, x') = exp(-γ||x - x'||²) with γ = 0.5. Apply it to a 2D dataset that forms two concentric circles and compare the first two principal components with linear PCA. Explain why kernel PCA can capture non-linear structure.

---

## Exercise 7: Support Vector Machines - Dual Formulation
**(5 marks)**

Consider the hard-margin SVM optimization problem and its dual formulation.

**Part 1 (2 marks):** Starting from the primal problem:
```
minimize: (1/2)||w||²
subject to: yᵢ(w^T xᵢ + b) ≥ 1, ∀i
```
Derive the dual problem using Lagrangian multipliers. Show all steps leading to:
```
maximize: Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢ^T xⱼ
subject to: Σᵢ αᵢyᵢ = 0, αᵢ ≥ 0
```

**Part 2 (1.5 marks):** For the 2D dataset:
- x₁ = [1, 1], y₁ = +1
- x₂ = [2, 2], y₂ = +1  
- x₃ = [-1, -1], y₃ = -1
- x₄ = [-2, -1], y₄ = -1

Set up the dual optimization problem matrices. Which points do you expect to be support vectors and why?

**Part 3 (1.5 marks):** Implement a simple SMO-style SVM solver in Python. Train on the above dataset and verify that your solution satisfies the KKT conditions. Plot the decision boundary and identify the support vectors.

---

## Exercise 8: Markov Decision Processes and Value Functions
**(4 marks)**

Consider a simple 3-state MDP with states S = {s₁, s₂, s₃}, actions A = {a₁, a₂}, discount factor γ = 0.9.

**Part 1 (1.5 marks):** Given the transition probabilities and rewards:
```
P(s₁|s₁,a₁) = 0.8, P(s₂|s₁,a₁) = 0.2, R(s₁,a₁) = 5
P(s₂|s₁,a₂) = 1.0, R(s₁,a₂) = 2
P(s₃|s₂,a₁) = 0.6, P(s₁|s₂,a₁) = 0.4, R(s₂,a₁) = 1
P(s₁|s₂,a₂) = 1.0, R(s₂,a₂) = 3
P(s₃|s₃,a₁) = P(s₃|s₃,a₂) = 1.0, R(s₃,a₁) = R(s₃,a₂) = 0
```

Write the Bellman equations for the value function V^π(s) under policy π(s₁) = a₁, π(s₂) = a₂, π(s₃) = a₁.

**Part 2 (1.5 marks):** Solve the system of Bellman equations to find V^π(s₁), V^π(s₂), V^π(s₃). Show your work step by step.

**Part 3 (1 mark):** Implement value iteration in Python for this MDP. Verify that it converges to the optimal value function and compare with your analytical solution for the given policy.

---

## Exercise 9: Policy Iteration vs Value Iteration
**(4 marks)**

Compare policy iteration and value iteration algorithms for solving MDPs.

**Part 1 (1.5 marks):** Describe the policy iteration algorithm step by step. Explain the policy evaluation and policy improvement phases. Under what conditions is the algorithm guaranteed to converge?

**Part 2 (1.5 marks):** For the 4×4 grid world with goal at (4,4) and obstacle at (2,2):
- Start state: (1,1)
- Actions: {up, down, left, right}
- Reward: -1 for each step, +10 for reaching goal
- γ = 0.95

Manually perform one complete iteration of policy iteration starting from the uniform random policy. Show the value function after policy evaluation and the improved policy.

**Part 3 (1 mark):** Implement both algorithms in Python and compare:
- Number of iterations to convergence
- Computational cost per iteration
- Final optimal policy (should be identical)

Plot the convergence curves and explain when you would prefer each algorithm.

---

## Exercise 10: Q-Learning and Temporal Difference Methods
**(5 marks)**

Consider the Q-learning algorithm for model-free reinforcement learning.

**Part 1 (2 marks):** Derive the Q-learning update rule from the Bellman equation for Q*. Explain why Q-learning is an off-policy method and how it differs from SARSA (on-policy). What is the role of the exploration parameter ε in ε-greedy policies?

**Part 2 (1.5 marks):** For a simple 2-state MDP with states {A, B}, actions {left, right}, and the following experience sequence:
```
(A, right, -1, B), (B, left, -1, A), (A, left, 0, A), (A, right, -1, B), (B, right, +10, terminal)
```

Initialize Q(s,a) = 0 for all state-action pairs. Using learning rate α = 0.5 and discount γ = 0.9, manually update the Q-table after each experience tuple. Show all calculations.

**Part 3 (1.5 marks):** Implement Q-learning for the classic FrozenLake environment (or a simple grid world). Plot the learning curve showing how the average episode return improves over training episodes. Experiment with different values of α and ε, and explain their effects on learning speed and final performance.

---

## Exercise 11: Regularization and Overfitting
**(4 marks)**

Consider Ridge and Lasso regularization for linear regression.

**Part 1 (1.5 marks):** For Ridge regression with loss function:
```
J(w) = (1/2N)Σᵢ(yᵢ - w^T xᵢ)² + λ||w||₂²
```

Derive the closed-form solution for the optimal weights. How does λ affect the solution compared to unregularized least squares?

**Part 2 (1.5 marks):** Given the design matrix X and target vector y:
```
X = [[1, 2], [1, -1], [1, 0]], y = [3, 0, 1]
```

Calculate the Ridge regression solution for λ = 1.0. Compare with the unregularized solution (λ = 0).

**Part 3 (1 mark):** Implement both Ridge and Lasso regression from scratch using gradient descent. Create a synthetic dataset where the true model is sparse (many coefficients are zero). Compare how well each method recovers the true sparsity pattern.

---

## Exercise 12: Feature Selection and Dimensionality Reduction
**(4 marks)**

Explore the relationship between PCA and feature selection.

**Part 1 (1.5 marks):** For the covariance matrix:
```
C = [[4, 2], [2, 1]]
```

Calculate the eigenvalues and eigenvectors. What percentage of variance is explained by the first principal component? If you project the data onto this component, what information is lost?

**Part 2 (1.5 marks):** Given data points in 2D: (1,1), (2,2), (3,3), (1,2), (2,3):
- Calculate the sample covariance matrix
- Find the principal components
- Project the data onto the first principal component
- Reconstruct the data and calculate the reconstruction error

**Part 3 (1 mark):** Implement both PCA and a simple filter-based feature selection method (e.g., correlation with target). Apply both to a high-dimensional dataset (e.g., digits recognition) and compare their effect on classification accuracy using a simple classifier.

---

## Grading Rubric

### Mathematical Derivations (40% of total marks)
- **Excellent (90-100%):** Complete, correct derivations with clear steps
- **Good (75-89%):** Mostly correct with minor algebraic errors
- **Satisfactory (60-74%):** Correct approach but with some mistakes
- **Needs Improvement (<60%):** Incorrect approach or major errors

### Numerical Calculations (30% of total marks)
- **Excellent:** All calculations correct with proper units/formatting
- **Good:** Minor computational errors that don't affect understanding
- **Satisfactory:** Some errors but demonstrates understanding
- **Needs Improvement:** Major computational mistakes

### Code Implementation (20% of total marks)
- **Excellent:** Clean, efficient, well-documented code that works correctly
- **Good:** Working code with minor inefficiencies or documentation issues
- **Satisfactory:** Code works but may be inefficient or poorly documented
- **Needs Improvement:** Code has bugs or doesn't address the problem

### Theoretical Understanding (10% of total marks)
- **Excellent:** Deep insights and connections between concepts
- **Good:** Good understanding with some insights
- **Satisfactory:** Basic understanding of concepts
- **Needs Improvement:** Limited understanding of underlying theory

---

## Study Tips for Exam Preparation

### Mathematical Preparation
1. **Practice derivations** repeatedly until you can do them without notes
2. **Understand the intuition** behind each mathematical concept
3. **Memorize key formulas** but focus on understanding their derivation
4. **Work through examples** step-by-step multiple times

### Coding Preparation
1. **Implement algorithms from scratch** to understand their mechanics
2. **Practice with different datasets** to test robustness
3. **Time yourself** to improve efficiency under exam conditions
4. **Debug systematically** and test edge cases

### Exam Strategy
1. **Read all questions first** and allocate time appropriately
2. **Start with easier parts** to build confidence
3. **Show all work** even if you're unsure of the final answer
4. **Check your answers** especially numerical calculations
5. **Manage your time** - don't spend too long on any single part

### Common Mistakes to Avoid
1. **Forgetting normalization constants** in probability derivations
2. **Matrix dimension errors** in linear algebra calculations
3. **Sign errors** in gradient calculations
4. **Overfitting to practice problems** instead of understanding concepts
5. **Not testing code** with simple examples first

---

*These exercises are designed to test both theoretical understanding and practical implementation skills. Practice regularly and focus on understanding the underlying concepts rather than memorizing solutions.*