# Practical Exercises and Projects 💻

This document provides detailed coding exercises, projects, and hands-on activities to complement your theoretical learning. Each exercise is designed to reinforce the concepts covered in the corresponding week's material.

> **📖 Looking for mathematical and theoretical exercises?** Check out [`MATHEMATICAL_EXERCISES.md`](MATHEMATICAL_EXERCISES.md) for derivations, proofs, and analytical problems that complement these practical coding exercises.

## 🛠️ Setup Instructions

### Required Software
```bash
# Python Environment (Recommended)
pip install numpy pandas matplotlib scikit-learn jupyter seaborn plotly

# Alternative R Environment
install.packages(c("caret", "ggplot2", "dplyr", "randomForest", "e1071"))
```

### Datasets for Practice
- **Iris Dataset:** Built-in classification dataset
- **Boston Housing:** Regression problems
- **Wine Quality:** Multi-class classification
- **Titanic:** Binary classification with missing data
- **MNIST:** Image classification (for advanced exercises)

---

## Week 1: Introduction to Machine Learning 🎯

### Exercise 1.1: Data Exploration Toolkit
**Objective:** Build a comprehensive data exploration pipeline

```python
# Template for Python implementation
class DataExplorer:
    def __init__(self, data):
        self.data = data
    
    def basic_stats(self):
        """Calculate and display basic statistics"""
        pass
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        pass
    
    def correlation_analysis(self):
        """Create correlation matrix and heatmap"""
        pass
    
    def distribution_plots(self):
        """Create histograms and box plots"""
        pass
```

**Tasks:**
- [ ] Load the Iris dataset
- [ ] Implement each method in the DataExplorer class
- [ ] Create visualizations for data distribution
- [ ] Identify potential data quality issues
- [ ] Generate a data quality report

**Expected Output:** A comprehensive report with statistics, visualizations, and insights about the dataset.

### Exercise 1.2: Train-Test Split Implementation
**Objective:** Understand data splitting strategies

```python
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Implement train-test split from scratch
    """
    # Your implementation here
    pass

def stratified_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Implement stratified split for classification
    """
    # Your implementation here
    pass
```

**Tasks:**
- [ ] Implement random train-test split
- [ ] Implement stratified split for classification
- [ ] Compare distributions between train and test sets
- [ ] Visualize the splitting results
- [ ] Test with different random seeds

### Exercise 1.3: Basic ML Pipeline
**Objective:** Create a complete ML workflow

**Tasks:**
- [ ] Create a pipeline class that handles: load → preprocess → train → evaluate
- [ ] Implement data normalization/standardization
- [ ] Add basic feature engineering capabilities
- [ ] Include model evaluation metrics
- [ ] Test with multiple datasets

**Deliverable:** A reusable ML pipeline that can handle different datasets and algorithms.

---

## Week 2: Linear Regression 📈

### Exercise 2.1: Linear Regression from Scratch
**Objective:** Implement linear regression using only NumPy

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        pass
    
    def predict(self, X):
        """Make predictions on new data"""
        pass
    
    def calculate_cost(self, X, y):
        """Calculate mean squared error"""
        pass
```

**Tasks:**
- [ ] Implement the `fit` method with gradient descent
- [ ] Implement prediction functionality
- [ ] Track cost function over iterations
- [ ] Visualize convergence behavior
- [ ] Compare with scikit-learn implementation

### Exercise 2.2: Advanced Gradient Descent
**Objective:** Explore different optimization techniques

**Tasks:**
- [ ] Implement batch gradient descent
- [ ] Implement stochastic gradient descent
- [ ] Implement mini-batch gradient descent
- [ ] Add momentum to gradient descent
- [ ] Compare convergence speeds and final results
- [ ] Visualize optimization paths

### Exercise 2.3: Polynomial Regression and Regularization
**Objective:** Handle non-linear relationships and overfitting

```python
class PolynomialRegression:
    def __init__(self, degree=2, regularization=None, lambda_reg=0.01):
        self.degree = degree
        self.regularization = regularization  # 'ridge' or 'lasso'
        self.lambda_reg = lambda_reg
    
    def create_polynomial_features(self, X):
        """Generate polynomial features"""
        pass
    
    def fit(self, X, y):
        """Train with regularization"""
        pass
```

**Tasks:**
- [ ] Implement polynomial feature generation
- [ ] Add Ridge regularization
- [ ] Add Lasso regularization
- [ ] Create learning curves to show overfitting
- [ ] Find optimal polynomial degree using validation

### Project 2: Housing Price Prediction
**Objective:** Complete regression project with real data

**Requirements:**
- [ ] Load and explore housing dataset
- [ ] Handle missing values and outliers
- [ ] Engineer meaningful features
- [ ] Implement multiple regression models
- [ ] Use cross-validation for model selection
- [ ] Create final prediction model
- [ ] Generate detailed performance report

**Deliverable:** A complete regression analysis with model comparison and insights.

---

## Week 3: Linear Classification 🎯

### Exercise 3.1: Logistic Regression Implementation
**Objective:** Build logistic regression from scratch

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        pass
    
    def fit(self, X, y):
        """Train using gradient descent"""
        pass
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        pass
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        pass
```

**Tasks:**
- [ ] Implement sigmoid function with numerical stability
- [ ] Implement gradient descent for logistic regression
- [ ] Add regularization options
- [ ] Handle multiclass classification (one-vs-rest)
- [ ] Compare with scikit-learn implementation

### Exercise 3.2: Classification Metrics Suite
**Objective:** Implement all classification evaluation metrics

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
    
    def confusion_matrix(self):
        """Calculate confusion matrix"""
        pass
    
    def accuracy(self):
        """Calculate accuracy"""
        pass
    
    def precision_recall_f1(self):
        """Calculate precision, recall, and F1-score"""
        pass
    
    def roc_curve_auc(self):
        """Calculate ROC curve and AUC"""
        pass
```

**Tasks:**
- [ ] Implement all basic metrics from scratch
- [ ] Create ROC and Precision-Recall curves
- [ ] Handle multiclass metrics (macro/micro averaging)
- [ ] Visualize all metrics with clear plots
- [ ] Compare threshold selection strategies

### Exercise 3.3: Perceptron Algorithm
**Objective:** Understand linear classification fundamentals

**Tasks:**
- [ ] Implement the perceptron algorithm
- [ ] Visualize decision boundary updates
- [ ] Handle non-linearly separable data
- [ ] Compare with logistic regression
- [ ] Apply to real classification problems

### Project 3: Medical Diagnosis Classifier
**Objective:** Build a complete classification system

**Requirements:**
- [ ] Use medical dataset (breast cancer, diabetes, etc.)
- [ ] Perform thorough exploratory data analysis
- [ ] Handle class imbalance appropriately
- [ ] Implement multiple classification algorithms
- [ ] Optimize decision thresholds for medical context
- [ ] Create comprehensive evaluation report
- [ ] Discuss ethical implications of false positives/negatives

---

## Week 4: Model Selection 🔧

### Exercise 4.1: Cross-Validation Implementation
**Objective:** Build robust model evaluation framework

```python
class CrossValidator:
    def __init__(self, k=5, strategy='kfold'):
        self.k = k
        self.strategy = strategy
    
    def split(self, X, y):
        """Generate train/validation indices"""
        pass
    
    def cross_validate(self, model, X, y, scoring='accuracy'):
        """Perform cross-validation"""
        pass
    
    def learning_curve(self, model, X, y, train_sizes):
        """Generate learning curves"""
        pass
```

**Tasks:**
- [ ] Implement k-fold cross-validation
- [ ] Implement stratified k-fold
- [ ] Implement leave-one-out CV
- [ ] Add time series cross-validation
- [ ] Create learning and validation curves

### Exercise 4.2: Hyperparameter Optimization
**Objective:** Automate model selection process

```python
class GridSearchCV:
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
    
    def fit(self, X, y):
        """Perform grid search"""
        pass
    
    def predict(self, X):
        """Use best model for predictions"""
        pass
```

**Tasks:**
- [ ] Implement grid search from scratch
- [ ] Add random search capability
- [ ] Implement Bayesian optimization (bonus)
- [ ] Create visualization of parameter space
- [ ] Compare different search strategies

### Exercise 4.3: Bias-Variance Analysis
**Objective:** Understand the bias-variance tradeoff

**Tasks:**
- [ ] Implement bias-variance decomposition
- [ ] Generate synthetic datasets with known functions
- [ ] Create bias-variance plots for different models
- [ ] Analyze effect of model complexity
- [ ] Demonstrate overfitting and underfitting

### Project 4: Model Selection Pipeline
**Objective:** Build automated model selection system

**Requirements:**
- [ ] Create pipeline that tests multiple algorithms
- [ ] Implement proper nested cross-validation
- [ ] Include feature selection techniques
- [ ] Add ensemble methods
- [ ] Generate comprehensive comparison report
- [ ] Include statistical significance testing
- [ ] Create reproducible results with proper random seeds

---

## Week 5: PAC Learning and Kernel Methods 🧠

### Exercise 5.1: VC Dimension Calculation
**Objective:** Understand theoretical learning concepts

**Tasks:**
- [ ] Calculate VC dimension for simple concept classes
- [ ] Implement empirical risk minimization
- [ ] Verify PAC bounds experimentally
- [ ] Analyze sample complexity for different problems
- [ ] Create visualizations of learning theory concepts

### Exercise 5.2: Kernel Functions Implementation
**Objective:** Master the kernel trick

```python
class KernelFunctions:
    @staticmethod
    def linear_kernel(X1, X2):
        """Linear kernel"""
        pass
    
    @staticmethod
    def polynomial_kernel(X1, X2, degree=3, coeff=1):
        """Polynomial kernel"""
        pass
    
    @staticmethod
    def rbf_kernel(X1, X2, gamma=1.0):
        """RBF (Gaussian) kernel"""
        pass
    
    @staticmethod
    def sigmoid_kernel(X1, X2, alpha=1.0, coeff=1.0):
        """Sigmoid kernel"""
        pass
```

**Tasks:**
- [ ] Implement all major kernel functions
- [ ] Verify kernel properties (symmetry, positive definiteness)
- [ ] Visualize kernel transformations
- [ ] Apply kernels to linear algorithms
- [ ] Compare kernel effects on different datasets

### Exercise 5.3: Kernel PCA Implementation
**Objective:** Apply kernels to dimensionality reduction

**Tasks:**
- [ ] Implement kernel PCA from scratch
- [ ] Compare with linear PCA
- [ ] Visualize non-linear dimensionality reduction
- [ ] Apply to real datasets
- [ ] Analyze computational complexity

---

## Week 6: Support Vector Machines ⚔️

### Exercise 6.1: SVM Implementation
**Objective:** Build SVM from mathematical principles

```python
class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None
    
    def fit(self, X, y):
        """Train SVM using SMO algorithm"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        pass
    
    def decision_function(self, X):
        """Calculate decision function values"""
        pass
```

**Tasks:**
- [ ] Implement simplified SMO algorithm
- [ ] Handle soft margin with C parameter
- [ ] Add kernel support
- [ ] Visualize decision boundaries and margins
- [ ] Identify and highlight support vectors

### Exercise 6.2: SVM Analysis and Comparison
**Objective:** Understand SVM behavior and performance

**Tasks:**
- [ ] Compare linear vs. kernel SVM
- [ ] Analyze effect of C parameter
- [ ] Study gamma parameter in RBF kernel
- [ ] Compare with other classification methods
- [ ] Handle multiclass classification

### Project 6: Image Classification with SVM
**Objective:** Apply SVM to computer vision task

**Requirements:**
- [ ] Use image dataset (digits, fashion, etc.)
- [ ] Implement feature extraction (HOG, LBP, etc.)
- [ ] Compare different kernels
- [ ] Optimize hyperparameters
- [ ] Evaluate on test set
- [ ] Create confusion matrix and error analysis

---

## Week 7: MDPs and Dynamic Programming 🎮

### Exercise 7.1: MDP Environment Implementation
**Objective:** Create interactive MDP environments

```python
class GridWorldMDP:
    def __init__(self, grid_size=(4, 4), start=(0, 0), goal=(3, 3)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.states = self._generate_states()
        self.actions = ['up', 'down', 'left', 'right']
        self.transition_probs = self._build_transition_model()
        self.rewards = self._build_reward_model()
    
    def step(self, state, action):
        """Take action in given state"""
        pass
    
    def get_possible_actions(self, state):
        """Get valid actions for state"""
        pass
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        pass
```

**Tasks:**
- [ ] Create grid world environment
- [ ] Add walls and obstacles
- [ ] Implement stochastic transitions
- [ ] Visualize environment and policies
- [ ] Create different reward structures

### Exercise 7.2: Value and Policy Iteration
**Objective:** Solve MDPs using dynamic programming

```python
class DynamicProgramming:
    def __init__(self, mdp, gamma=0.9, theta=1e-6):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
    
    def policy_evaluation(self, policy):
        """Evaluate given policy"""
        pass
    
    def policy_improvement(self, V):
        """Improve policy based on value function"""
        pass
    
    def policy_iteration(self):
        """Full policy iteration algorithm"""
        pass
    
    def value_iteration(self):
        """Value iteration algorithm"""
        pass
```

**Tasks:**
- [ ] Implement policy evaluation
- [ ] Implement policy improvement
- [ ] Implement full policy iteration
- [ ] Implement value iteration
- [ ] Compare convergence rates
- [ ] Visualize value functions and policies

### Project 7: Game AI with MDPs
**Objective:** Create intelligent game-playing agent

**Requirements:**
- [ ] Choose game (Tic-Tac-Toe, Connect-4, etc.)
- [ ] Model as MDP
- [ ] Implement optimal policy
- [ ] Create human vs. AI interface
- [ ] Analyze different reward structures
- [ ] Compare with other game AI approaches

---

## Week 8: Reinforcement Learning 🤖

### Exercise 8.1: Q-Learning Implementation
**Objective:** Master temporal difference learning

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        pass
    
    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule"""
        pass
    
    def decay_epsilon(self):
        """Decrease exploration rate"""
        pass
```

**Tasks:**
- [ ] Implement Q-learning algorithm
- [ ] Add epsilon-greedy exploration
- [ ] Implement epsilon decay
- [ ] Track learning progress
- [ ] Visualize Q-values and policies

### Exercise 8.2: SARSA Implementation
**Objective:** Compare on-policy vs off-policy learning

**Tasks:**
- [ ] Implement SARSA algorithm
- [ ] Compare with Q-learning
- [ ] Analyze convergence behavior
- [ ] Test on different environments
- [ ] Study exploration vs. exploitation tradeoff

### Exercise 8.3: Function Approximation
**Objective:** Handle large state spaces

**Tasks:**
- [ ] Implement Q-learning with function approximation
- [ ] Use neural networks for value function
- [ ] Compare with tabular methods
- [ ] Handle continuous state spaces
- [ ] Analyze stability issues

### Final Project: Autonomous Agent
**Objective:** Build complete RL application

**Requirements:**
- [ ] Choose complex environment (CartPole, Mountain Car, etc.)
- [ ] Implement multiple RL algorithms
- [ ] Compare performance and learning curves
- [ ] Add visualization of agent behavior
- [ ] Analyze hyperparameter sensitivity
- [ ] Create comprehensive evaluation report
- [ ] Discuss potential real-world applications

---

## 📊 Assessment Rubric

### Technical Implementation (40%)
- Code correctness and efficiency
- Proper use of algorithms and data structures
- Error handling and edge cases
- Code documentation and style

### Understanding and Analysis (30%)
- Correct interpretation of results
- Insightful analysis of algorithm behavior
- Understanding of theoretical concepts
- Ability to explain trade-offs

### Experimentation and Validation (20%)
- Appropriate experimental design
- Proper use of evaluation metrics
- Statistical significance testing
- Comparison with baselines

### Communication and Documentation (10%)
- Clear explanation of methodology
- Well-structured reports
- Effective visualizations
- Professional presentation

---

## 🎯 Success Tips

1. **Start Simple:** Always implement the basic version first
2. **Test Incrementally:** Test each component before integration
3. **Visualize Everything:** Create plots to understand algorithm behavior
4. **Compare Baselines:** Always compare with existing implementations
5. **Document Learning:** Keep notes on insights and challenges
6. **Experiment Freely:** Try different parameters and datasets
7. **Seek Patterns:** Look for common themes across algorithms
8. **Practice Regularly:** Consistent daily practice beats weekend marathons

---

*Complete these exercises progressively to build strong practical ML skills!*