# Advanced Machine Learning Exercises 🎓

This document contains advanced exercises designed to challenge students and prepare them for comprehensive machine learning examinations. These exercises build upon the fundamental concepts and explore deeper theoretical and practical aspects.

---

## Exercise 13: Kernel SVM with Soft Margin
**(6 marks)**

Consider the soft-margin SVM with kernel functions.

**Part 1 (2 marks):** Starting from the soft-margin primal problem:
```
minimize: (1/2)||w||² + C∑ᵢ ξᵢ
subject to: yᵢ(w^T φ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

Derive the dual formulation and show how the kernel trick allows us to avoid explicit computation of φ(x). What is the role of the parameter C?

**Part 2 (2 marks):** Given the RBF kernel k(x,x') = exp(-γ||x-x'||²) and the training data:
- x₁ = [0, 0], y₁ = +1
- x₂ = [1, 1], y₂ = +1  
- x₃ = [0, 1], y₃ = -1
- x₄ = [1, 0], y₄ = -1

With γ = 1.0, compute the Gram matrix K where Kᵢⱼ = k(xᵢ, xⱼ). Explain why this problem is not linearly separable in the original space but may be separable in the kernel space.

**Part 3 (2 marks):** Implement kernel SVM using the dual formulation. Compare the decision boundaries produced by linear, polynomial (degree 2), and RBF kernels on a 2D dataset with overlapping classes. Analyze how the choice of kernel parameters affects the complexity of the decision boundary.

---

## Exercise 14: Ensemble Methods and Boosting
**(5 marks)**

Explore the theoretical foundations of ensemble learning.

**Part 1 (2 marks):** For a binary classification ensemble with L weak learners, where each learner hₗ(x) outputs ±1, prove that if each learner has error rate εₗ < 0.5 and the learners are independent, then the majority vote ensemble has error rate:
```
P(ensemble error) = P(∑ᵢ I(hᵢ(x) ≠ y) > L/2)
```

Show that this error rate decreases as L increases (under the independence assumption).

**Part 2 (1.5 marks):** In AdaBoost, the weight update rule for misclassified examples is:
```
w⁽ᵗ⁺¹⁾ᵢ = w⁽ᵗ⁾ᵢ × exp(αₜ)
```
where αₜ = (1/2)ln((1-εₜ)/εₜ). Derive this update rule from the perspective of minimizing an exponential loss function. What happens to αₜ when εₜ → 0 and when εₜ → 0.5?

**Part 3 (1.5 marks):** Implement AdaBoost with decision stumps (one-level decision trees) as weak learners. Train on a 2D dataset and visualize how the decision boundary evolves as you add more weak learners. Compare the final accuracy with a single decision tree and analyze the effect of ensemble size on bias and variance.

---

## Exercise 15: Gaussian Mixture Models and EM Algorithm
**(6 marks)**

Study unsupervised learning with probabilistic models.

**Part 1 (2.5 marks):** For a Gaussian Mixture Model with K components:
```
p(x) = ∑ₖ πₖ N(x|μₖ, Σₖ)
```

Derive the EM algorithm update equations for the parameters {πₖ, μₖ, Σₖ}. Start from the log-likelihood function and introduce the latent variable zᵢₖ indicating which component generated point xᵢ.

**Part 2 (1.5 marks):** Given 1D data points: [1, 2, 8, 9, 10] and assuming K=2 Gaussian components with equal variances σ² = 1, perform one complete E-M iteration starting from:
- π₁ = π₂ = 0.5
- μ₁ = 3, μ₂ = 7

Calculate the responsibilities γᵢₖ in the E-step and the updated parameters in the M-step.

**Part 3 (2 marks):** Implement the EM algorithm for GMM from scratch. Apply it to a 2D dataset with 3 natural clusters and compare with K-means clustering. Discuss the advantages of GMM over K-means in terms of cluster shape assumptions and probabilistic interpretation.

---

## Exercise 16: Hidden Markov Models and Viterbi Algorithm
**(5 marks)**

Analyze sequence modeling with HMMs.

**Part 1 (2 marks):** For an HMM with states S = {s₁, s₂} and observations O = {o₁, o₂}, given:
```
Transition: A = [[0.7, 0.3], [0.4, 0.6]]
Emission: B = [[0.9, 0.1], [0.2, 0.8]]
Initial: π = [0.6, 0.4]
```

Calculate the probability of the observation sequence [o₁, o₂, o₁] using the forward algorithm. Show the forward probability matrix step by step.

**Part 2 (1.5 marks):** Implement the Viterbi algorithm for the same HMM. Find the most likely state sequence that generated the observation sequence [o₁, o₂, o₁]. Compare this with the forward probabilities from Part 1.

**Part 3 (1.5 marks):** Apply your HMM implementation to a practical problem: modeling a simple weather system where hidden states are {Sunny, Rainy} and observations are {Dry, Wet, Soggy}. Generate synthetic data and test whether the Viterbi algorithm can correctly recover the hidden weather states.

---

## Exercise 17: Monte Carlo Methods and MCMC
**(4 marks)**

Explore sampling-based inference methods.

**Part 1 (1.5 marks):** Explain the Metropolis-Hastings algorithm for sampling from a target distribution π(x). Given a proposal distribution q(x'|x), derive the acceptance probability:
```
α(x → x') = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
```

What happens when q is symmetric (q(x'|x) = q(x|x'))?

**Part 2 (1.5 marks):** Use the Metropolis algorithm to sample from a 2D Gaussian distribution N([0,0], [[1,0.8],[0.8,2]]). Implement with a symmetric proposal distribution (e.g., Gaussian random walk) and analyze:
- Convergence of sample mean to true mean
- Effect of proposal variance on acceptance rate and mixing
- Autocorrelation in the sample chain

**Part 3 (1 mark):** Apply MCMC to estimate the parameters of a Bayesian linear regression model. Compare the posterior distributions obtained through sampling with the analytical solution for a simple 1D regression problem.

---

## Exercise 18: Information Theory and Mutual Information
**(4 marks)**

Connect information theory concepts to machine learning.

**Part 1 (1.5 marks):** For discrete random variables X and Y, prove that:
```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

where I(X;Y) is mutual information and H(·) is entropy. Explain the intuitive meaning of mutual information in the context of feature selection.

**Part 2 (1.5 marks):** Calculate the mutual information between features and the target for the following dataset:
```
Feature X: [0, 0, 1, 1, 0, 1]
Target Y:  [0, 0, 0, 1, 1, 1]
```

Show all entropy calculations. Would you select this feature based on its mutual information score?

**Part 3 (1 mark):** Implement mutual information-based feature selection and apply it to a real dataset (e.g., wine quality). Compare the selected features with those chosen by correlation-based methods. Explain why mutual information can capture non-linear relationships that correlation cannot.

---

## Exercise 19: Bayesian Neural Networks
**(5 marks)**

Explore uncertainty quantification in deep learning.

**Part 1 (2 marks):** In a Bayesian neural network, instead of point estimates for weights w, we maintain a posterior distribution p(w|D). Explain how:
- Prior beliefs about weights are incorporated
- Predictive uncertainty is quantified through posterior integration
- The connection to dropout as an approximate inference technique

**Part 2 (1.5 marks):** For a simple 1-hidden-layer network with 2 weights {w₁, w₂}, assume Gaussian priors w₁ ~ N(0,1), w₂ ~ N(0,1) and likelihood p(y|x,w) = N(w₁x + w₂, σ²). Given data point (x=1, y=2), derive the posterior distribution (assuming σ² is known).

**Part 3 (1.5 marks):** Implement Monte Carlo dropout to approximate Bayesian inference in a neural network. Train on a small regression dataset and plot prediction intervals showing epistemic uncertainty. Compare with a standard neural network's point predictions.

---

## Exercise 20: Advanced Reinforcement Learning - Policy Gradients
**(6 marks)**

Study policy-based reinforcement learning methods.

**Part 1 (2.5 marks):** Derive the policy gradient theorem. Starting from the objective J(θ) = E_π[∑ₜγᵗr(sₜ,aₜ)] where π(a|s,θ) is a parameterized policy, show that:
```
∇_θ J(θ) = E_π[∑ₜ ∇_θ log π(aₜ|sₜ,θ) · Qᵖⁱ(sₜ,aₜ)]
```

Explain the intuition behind this result and why it's called REINFORCE.

**Part 2 (1.5 marks):** For a simple 2-state MDP with softmax policy:
```
π(a|s,θ) = exp(θₛₐ) / ∑ₐ' exp(θₛₐ')
```

Given the state-action values:
- Q(s₁,a₁) = 5, Q(s₁,a₂) = 2
- Q(s₂,a₁) = 1, Q(s₂,a₂) = 4

Calculate the policy gradient for each parameter θₛₐ when visiting state s₁.

**Part 3 (2 marks):** Implement the REINFORCE algorithm for a simple environment (e.g., CartPole or a custom grid world). Compare its learning curve with Q-learning. Discuss the advantages and disadvantages of policy gradient methods compared to value-based methods like Q-learning.

---

## Exercise 21: Multi-Armed Bandits and Exploration
**(4 marks)**

Analyze the exploration-exploitation tradeoff.

**Part 1 (1.5 marks):** For the ε-greedy strategy in multi-armed bandits, derive the regret bound. If arm i has true mean μᵢ and we define Δᵢ = μ* - μᵢ where μ* is the optimal arm's mean, show that the expected regret after T rounds is:
```
E[Regret(T)] ≤ ∑ᵢ:Δᵢ>0 (εT/K + (1+ε)ln(T)/Δᵢ)
```

**Part 2 (1.5 marks):** Compare ε-greedy with Upper Confidence Bound (UCB) strategy. For a 3-armed bandit with true means [0.7, 0.5, 0.3], simulate 1000 rounds and plot the cumulative regret for both strategies. Which performs better and why?

**Part 3 (1 mark):** Implement Thompson Sampling for Bernoulli bandits using Beta priors. Compare its performance with ε-greedy and UCB on the same 3-armed bandit problem. Explain why Thompson Sampling often outperforms other strategies in practice.

---

## Exercise 22: Generative Adversarial Networks (Theory)
**(5 marks)**

Study the theoretical foundations of GANs.

**Part 1 (2 marks):** In the GAN framework, the generator G and discriminator D play a minimax game:
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1-D(G(z)))]
```

Prove that the optimal discriminator D* for a fixed generator G is:
```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

where p_data is the true data distribution and p_g is the generator's distribution.

**Part 2 (1.5 marks):** Substitute the optimal discriminator back into the objective function and show that minimizing over G is equivalent to minimizing the Jensen-Shannon divergence between p_data and p_g. What does this imply about the global optimum?

**Part 3 (1.5 marks):** Explain the mode collapse problem in GANs and discuss why it occurs from a theoretical perspective. Implement a simple 1D GAN to demonstrate mode collapse on a mixture of Gaussians and show how techniques like mini-batch discrimination can help.

---

## Exercise 23: Transfer Learning and Domain Adaptation
**(4 marks)**

Explore knowledge transfer between different domains.

**Part 1 (1.5 marks):** Define the domain adaptation problem formally. Given source domain DS = {XS, P(XS), P(YS|XS)} and target domain DT = {XT, P(XT), P(YT|XT)}, explain the different types of domain shift:
- Covariate shift: P(XS) ≠ P(XT)
- Prior shift: P(YS) ≠ P(YT)  
- Concept shift: P(YS|XS) ≠ P(YT|XT)

**Part 2 (1.5 marks):** For covariate shift with importance weighting, derive the re-weighted empirical risk:
```
R̂(h) = (1/n) ∑ᵢ w(xᵢ) L(h(xᵢ), yᵢ)
```

where w(x) = P(XT)/P(XS). How would you estimate these weights in practice?

**Part 3 (1 mark):** Implement a simple transfer learning experiment: train a CNN on CIFAR-10, then fine-tune on a subset of CIFAR-100. Compare the performance with training from scratch on CIFAR-100. Analyze which layers transfer better and why.

---

## Exercise 24: Causal Inference and Confounding
**(5 marks)**

Connect causality concepts to machine learning.

**Part 1 (2 marks):** Explain the difference between correlation and causation using directed acyclic graphs (DAGs). For the relationship X → Z ← Y, show why conditioning on Z can create spurious correlation between X and Y even when they are marginally independent.

**Part 2 (1.5 marks):** Given the causal graph: Treatment T → Outcome Y ← Confounder C → T, derive the adjustment formula for estimating the causal effect of T on Y:
```
P(Y|do(T=t)) = ∑_c P(Y|T=t,C=c)P(C=c)
```

Explain why simply comparing P(Y|T=1) and P(Y|T=0) would give biased estimates.

**Part 3 (1.5 marks):** Implement propensity score matching to estimate causal effects from observational data. Use a synthetic dataset where you know the true causal effect and compare your estimate with the naive comparison of treatment groups. Discuss the assumptions required for causal identification.

---

## Final Comprehensive Exercise: End-to-End ML Project
**(10 marks)**

Design and implement a complete machine learning solution.

**Problem Statement:** You are given a real-world dataset of your choice (e.g., predicting house prices, medical diagnosis, customer churn). Develop a comprehensive ML solution that demonstrates mastery of multiple concepts covered in the course.

**Requirements:**

**Part 1: Data Analysis and Preprocessing (2 marks)**
- Perform thorough exploratory data analysis
- Handle missing values, outliers, and data quality issues
- Apply appropriate feature engineering techniques
- Document your decisions and justify your choices

**Part 2: Model Development (3 marks)**
- Implement at least 3 different learning algorithms from scratch
- Apply proper cross-validation for model selection
- Tune hyperparameters systematically
- Handle class imbalance or other data-specific challenges

**Part 3: Evaluation and Interpretation (2 marks)**
- Use appropriate evaluation metrics for your problem
- Perform statistical significance testing
- Analyze feature importance and model interpretability
- Assess potential biases and fairness concerns

**Part 4: Advanced Techniques (2 marks)**
- Implement ensemble methods or regularization
- Apply dimensionality reduction if appropriate
- Consider uncertainty quantification
- Discuss computational complexity and scalability

**Part 5: Communication and Documentation (1 mark)**
- Create clear visualizations and explanations
- Write a technical report summarizing your approach
- Present limitations and future work
- Provide reproducible code with proper documentation

**Grading Criteria:**
- Technical correctness and implementation quality
- Appropriate choice and application of methods
- Depth of analysis and insights generated
- Quality of communication and documentation
- Creativity and going beyond basic requirements

---

## Study Guide for Advanced Topics

### Key Theoretical Concepts to Master
1. **Optimization Theory:** Convexity, gradient methods, constrained optimization
2. **Probability Theory:** Bayesian inference, information theory, sampling methods
3. **Statistical Learning Theory:** PAC learning, VC dimension, generalization bounds
4. **Linear Algebra:** Eigendecomposition, matrix calculus, dimensionality reduction
5. **Graph Theory:** Probabilistic graphical models, causal graphs

### Implementation Skills
1. **Algorithm Implementation:** Build methods from mathematical foundations
2. **Numerical Stability:** Handle floating-point precision and numerical issues
3. **Computational Efficiency:** Understand algorithmic complexity and optimization
4. **Software Engineering:** Write modular, testable, and maintainable code
5. **Experimental Design:** Proper evaluation protocols and statistical testing

### Problem-Solving Approach
1. **Mathematical Rigor:** Show complete derivations and proofs
2. **Empirical Validation:** Test theoretical predictions with experiments
3. **Critical Analysis:** Question assumptions and limitations
4. **Creative Solutions:** Combine multiple techniques innovatively
5. **Practical Considerations:** Address real-world constraints and requirements

---

*These advanced exercises challenge students to integrate multiple ML concepts and develop deep understanding of both theoretical foundations and practical applications.*