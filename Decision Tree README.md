# 🌳 Decision Tree — Machine Learning

A complete beginner-to-interview guide to **Decision Tree Machine Learning**, including theory, intuition, mathematical concepts, algorithms, Python implementation, visualization, pruning, overfitting, feature importance, Decision Tree Regression, and interview questions.

---

# 📌 Table of Contents

1. [What is a Decision Tree?](#-what-is-a-decision-tree)
2. [Real-Life Example](#-real-life-example)
3. [Decision Tree Structure](#-decision-tree-structure)
4. [How Does a Decision Tree Work?](#-how-does-a-decision-tree-work)
5. [Important Terminology](#-important-terminology)
6. [Classification vs Regression](#-classification-vs-regression)
7. [Decision Tree Classification](#-decision-tree-classification)
8. [Decision Tree Regression](#-decision-tree-regression)
9. [Impurity](#-impurity)
10. [Entropy](#-entropy)
11. [Information Gain](#-information-gain)
12. [Gini Impurity](#-gini-impurity)
13. [Entropy vs Gini](#-entropy-vs-gini)
14. [Gain Ratio](#-gain-ratio)
15. [Variance Reduction](#-variance-reduction)
16. [ID3](#-id3)
17. [C4.5](#-c45)
18. [CART](#-cart)
19. [Decision Tree Algorithm Comparison](#-decision-tree-algorithm-comparison)
20. [How the Best Split is Selected](#-how-the-best-split-is-selected)
21. [Continuous Variables](#-continuous-variables)
22. [Categorical Variables](#-categorical-variables)
23. [Overfitting](#-overfitting)
24. [Pruning](#-pruning)
25. [Pre-Pruning](#-pre-pruning)
26. [Post-Pruning](#-post-pruning)
27. [Important Hyperparameters](#-important-hyperparameters)
28. [Feature Importance](#-feature-importance)
29. [Advantages](#-advantages)
30. [Disadvantages](#-disadvantages)
31. [Decision Tree vs Random Forest](#-decision-tree-vs-random-forest)
32. [Python Classification Example](#-python-classification-example)
33. [Decision Tree Visualization](#-decision-tree-visualization)
34. [Prediction Example](#-prediction-example)
35. [Decision Tree Regression Example](#-decision-tree-regression-example)
36. [Model Evaluation](#-model-evaluation)
37. [Complete ML Workflow](#-complete-ml-workflow)
38. [Important Interview Questions](#-important-interview-questions)
39. [Advanced Interview Questions](#-advanced-interview-questions)
40. [Practical Interview Scenario](#-practical-interview-scenario)
41. [Quick Revision](#-quick-revision)

---

# 🌳 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm used for:

- Classification
- Regression

It works by repeatedly asking questions about the input features and splitting the data into smaller groups.

The final prediction is made at a **leaf node**.

### Simple idea

Suppose we want to decide whether a customer will buy a product.

The tree may ask:

```text
Is Age > 30?
       |
   Yes | No
       |
       v
Is Income > 50K?
       |
   Yes | No
       |
       v
     Buy?
```

The Decision Tree converts the learning problem into a series of **IF-THEN rules**.

Example:

```text
IF Age > 30
AND Income > 50,000
THEN Customer = Buy
```

---

# 🧠 Real-Life Example

Imagine a bank wants to decide whether a customer is likely to:

```text
Approve Loan
     |
     v
Credit Score > 700?
    /        \
  Yes        No
   |          |
Income > 50K?  Reject
 /      \
Yes      No
 |        |
Approve   Reject
```

The model has learned rules from historical customer data.

---

# 🌲 Decision Tree Structure

A Decision Tree contains several important parts.

## 1. Root Node

The first/top node.

Example:

```text
Credit Score > 700?
```

The root contains the first major decision.

---

## 2. Internal Node

A node where another decision is made.

Example:

```text
Income > 50K?
```

---

## 3. Branch

The connection between nodes.

Example:

```text
Yes
No
```

---

## 4. Leaf Node

The final output/prediction.

Example:

```text
Approve
Reject
```

---

## 5. Depth

Depth represents how many levels exist from the root to the deepest leaf.

Example:

```text
                 Root
                   |
             Credit Score
              /          \
            Yes           No
            |              |
         Income          Reject
        /     \
      Yes      No
      |         |
   Approve    Reject
```

The deepest path determines the tree depth.

---

# ⚙️ How Does a Decision Tree Work?

The basic process is:

```text
              Dataset
                 |
                 v
        Select Best Feature
                 |
                 v
            Split Data
              /    \
             /      \
            v        v
       Subset 1   Subset 2
          |           |
          v           v
    Select Best   Select Best
      Feature       Feature
          |           |
          v           v
       Continue Recursively
                 |
                 v
              Leaf Node
                 |
                 v
             Prediction
```

The tree repeatedly finds the split that produces the best separation of the target classes or best reduction in regression error.

Scikit-learn describes this as recursively partitioning the feature space and stopping according to conditions such as `max_depth`, `min_samples_split`, or `min_impurity_decrease`.

---

# 🔑 Important Terminology

| Term | Meaning |
|---|---|
| Root | First node |
| Internal Node | Decision node |
| Branch | Connection between nodes |
| Leaf | Final prediction |
| Depth | Maximum number of levels |
| Split | Dividing data into groups |
| Parent Node | Node before splitting |
| Child Node | Node created after splitting |
| Impurity | Amount of mixture/uncertainty |
| Pure Node | Node containing mostly/only one class |
| Pruning | Removing unnecessary branches |

---

# 🔀 Classification vs Regression

Decision Trees can solve two major problems.

## Classification

Target is a category/class.

Examples:

```text
Spam / Not Spam
Yes / No
Fraud / Not Fraud
Approved / Rejected
Disease / No Disease
```

Use:

```python
DecisionTreeClassifier
```

---

## Regression

Target is a continuous numerical value.

Examples:

```text
House Price
Salary
Sales
Temperature
Revenue
```

Use:

```python
DecisionTreeRegressor
```

---

# 🌳 Decision Tree Classification

Example dataset:

| Age | Income | Buy |
|---:|---:|---|
| 22 | 25000 | No |
| 25 | 30000 | No |
| 35 | 60000 | Yes |
| 40 | 80000 | Yes |
| 45 | 90000 | Yes |

Features:

```text
Age
Income
```

Target:

```text
Buy
```

The Decision Tree tries to find the best feature and threshold.

For example:

```text
Income <= 50000?
       /       \
     Yes       No
      |         |
     No        Yes
```

---

# 📊 Impurity

Impurity tells us how mixed the classes are inside a node.

Example:

```text
Node A:

Yes = 5
No  = 0
```

This node is completely pure.

```text
Impurity = 0
```

But:

```text
Node B:

Yes = 5
No = 5
```

This node is highly mixed.

Therefore:

```text
High impurity = mixed classes

Low impurity = mostly one class

Zero impurity = completely pure
```

Two important impurity measures are:

1. Entropy
2. Gini Impurity

---

# 📐 Entropy

Entropy measures uncertainty or disorder.

Formula:

```text
Entropy(S) = - Σ p(i) log2(p(i))
```

For binary classification:

```text
Entropy = -p1 log2(p1) - p2 log2(p2)
```

where:

```text
p1 = probability of class 1

p2 = probability of class 2
```

---

## Example

Suppose:

```text
Yes = 5
No = 5
```

Therefore:

```text
P(Yes) = 0.5

P(No) = 0.5
```

Entropy:

```text
Entropy
= -(0.5 × log2(0.5))
  -(0.5 × log2(0.5))

= 1
```

Maximum entropy for binary classification is:

```text
1
```

---

## Pure Node

Suppose:

```text
Yes = 10
No = 0
```

Then:

```text
P(Yes) = 1
P(No) = 0
```

Entropy:

```text
0
```

So:

```text
Pure node → Entropy = 0
Mixed node → Higher entropy
```

---

# 📈 Information Gain

Information Gain tells us how much uncertainty is reduced after splitting.

Formula:

```text
Information Gain
=
Entropy(parent)
-
Weighted Entropy(children)
```

More formally:

```text
IG(S,A)
=
H(S)
-
Σ (|Sv| / |S|) H(Sv)
```

The best split generally has:

```text
Highest Information Gain
```

---

# 🧮 Simple Information Gain Example

Suppose parent node:

```text
10 records

Yes = 5
No = 5
```

Parent entropy:

```text
1
```

After splitting:

```text
Left:
Yes = 4
No = 0

Right:
Yes = 1
No = 5
```

Left node:

```text
Entropy = 0
```

Right node:

```text
Entropy ≈ 0.65
```

Weighted child entropy:

```text
(4/10 × 0) + (6/10 × 0.65)

≈ 0.39
```

Therefore:

```text
Information Gain
= 1 - 0.39

≈ 0.61
```

A split with higher Information Gain is generally preferred by entropy-based tree algorithms.

---

# 🟢 Gini Impurity

Gini impurity measures how often a randomly selected observation would be incorrectly classified if it were randomly assigned according to the class distribution.

Formula:

```text
Gini = 1 - Σ p(i)^2
```

For binary classification:

```text
Gini = 1 - (p1² + p2²)
```

---

## Example

Suppose:

```text
Yes = 4
No = 6
```

Then:

```text
P(Yes) = 0.4

P(No) = 0.6
```

Gini:

```text
Gini
= 1 - (0.4² + 0.6²)

= 1 - (0.16 + 0.36)

= 1 - 0.52

= 0.48
```

Therefore:

```text
Gini = 0.48
```

---

# ⚖️ Entropy vs Gini

| Feature | Entropy | Gini |
|---|---|---|
| Measures | Uncertainty | Impurity |
| Formula | `-Σp log2(p)` | `1 - Σp²` |
| Range | 0 to 1 for binary | 0 to 0.5 for binary |
| Log calculation | Yes | No |
| Computational cost | Slightly higher | Slightly lower |
| Common in sklearn | Yes | Yes |
| Default sklearn criterion | No | Yes |

Current scikit-learn supports `gini`, `entropy`, and `log_loss` for `DecisionTreeClassifier`, with `gini` as the default criterion.

---

# 🎯 Gain Ratio

Information Gain can have a problem.

A feature with many unique values can sometimes receive an artificially high Information Gain.

C4.5 introduced **Gain Ratio** to reduce this bias.

Formula:

```text
Gain Ratio
=
Information Gain
/
Split Information
```

C4.5 uses Gain Ratio rather than raw Information Gain.

---

# 📉 Variance Reduction

For Decision Tree Regression, the target is numerical.

Instead of classification impurity, the algorithm tries to reduce the variation/error inside the child nodes.

Common measures include:

```text
MSE
Variance Reduction
```

Example:

```text
House Price

50K
55K
52K
200K
210K
220K
```

A good split should create groups with similar prices.

---

# 🧠 ID3

ID3 stands for:

```text
Iterative Dichotomiser 3
```

It is one of the early Decision Tree algorithms.

Main characteristics:

```text
Uses Entropy
Uses Information Gain
Primarily categorical attributes
Historically creates multi-way splits
```

Basic process:

```text
Calculate Entropy
        ↓
Calculate Information Gain
        ↓
Select highest Information Gain
        ↓
Split Dataset
        ↓
Repeat
```

ID3 is important historically and conceptually, although modern production implementations often use other tree algorithms.

---

# 🌲 C4.5

C4.5 is an extension of ID3.

Important improvements include:

```text
Uses Gain Ratio
Handles continuous attributes
Supports more practical tree construction
Includes pruning
```

Main idea:

```text
ID3
 |
 +-- Entropy
 +-- Information Gain

C4.5
 |
 +-- Entropy
 +-- Gain Ratio
 +-- Continuous values
 +-- Pruning
```

---

# 🌳 CART

CART stands for:

```text
Classification And Regression Trees
```

CART is extremely important because scikit-learn's Decision Tree implementation is based on an optimized CART approach.

CART supports:

```text
Classification
Regression
```

Typical criteria:

### Classification

```text
Gini
```

### Regression

```text
MSE
```

CART creates **binary splits**.

Example:

```text
Age <= 30?
   /    \
 Yes     No
```

Rather than:

```text
Age = 20
Age = 30
Age = 40
Age = 50
```

---

# 🆚 Decision Tree Algorithm Comparison

| Algorithm | Split Criterion | Classification | Regression | Split Type |
|---|---|---:|---:|---|
| ID3 | Information Gain | Yes | No | Multi-way |
| C4.5 | Gain Ratio | Yes | No | Multi-way |
| CART | Gini / MSE | Yes | Yes | Binary |

Scikit-learn specifically uses an optimized CART implementation.

---

# 🔍 How the Best Split is Selected

Suppose we have:

```text
Age
Income
Credit Score
```

The algorithm evaluates possible splits.

For example:

```text
Age <= 25
Age <= 30
Age <= 35

Income <= 30000
Income <= 50000
Income <= 70000

Credit Score <= 600
Credit Score <= 700
Credit Score <= 750
```

Each candidate split is evaluated.

For classification:

```text
Gini
or
Entropy / Information Gain
```

For regression:

```text
MSE / variance reduction
```

The best split is selected according to the chosen criterion.

---

# 🔢 Continuous Variables

Decision Trees can split numerical features using thresholds.

Example:

```text
Age <= 30
```

Data:

```text
Age
22
25
28
31
35
40
```

Possible thresholds can be evaluated.

Example:

```text
Age <= 30
```

creates:

```text
Left:
22
25
28

Right:
31
35
40
```

---

# 🏷️ Categorical Variables

Example:

```text
Gender
Male
Female
```

or:

```text
City
Kolkata
Delhi
Mumbai
Hyderabad
```

Depending on the implementation, categorical data may need encoding before passing it to the model.

A common approach in scikit-learn workflows is:

```text
Categorical Feature
        ↓
Encoding
        ↓
Numerical Representation
        ↓
Decision Tree
```

For example:

```python
from sklearn.preprocessing import OneHotEncoder
```

---

# ⚠️ Overfitting

Decision Trees are highly prone to overfitting when allowed to grow too deeply.

Example:

```text
Training Accuracy = 100%
Testing Accuracy  = 72%
```

This indicates possible overfitting.

A very large tree may memorize training data rather than learn general patterns.

Example:

```text
                Root
                  |
             Feature A
            /         \
       Feature B     Feature C
       /    \         /     \
      ...   ...      ...    ...
```

If the tree becomes extremely deep, it may capture noise.

---

# ✂️ Pruning

Pruning means reducing unnecessary branches from a Decision Tree.

Goal:

```text
Smaller Tree
     ↓
Less Complexity
     ↓
Less Overfitting
     ↓
Better Generalization
```

There are two major concepts:

```text
Pre-Pruning
Post-Pruning
```

---

# 🛑 Pre-Pruning

Stop the tree from becoming too large during training.

Important parameters:

```python
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
```

Example:

```python
DecisionTreeClassifier(
    max_depth=5
)
```

---

# ✂️ Post-Pruning

First grow a larger tree and then remove branches that do not improve generalization.

One important method is:

```text
Cost-Complexity Pruning
```

In scikit-learn this is controlled using:

```python
ccp_alpha
```

Example:

```python
DecisionTreeClassifier(
    ccp_alpha=0.01
)
```

Scikit-learn documents `ccp_alpha` as the parameter for Minimal Cost-Complexity Pruning.

---

# 🎛️ Important Hyperparameters

## 1. max_depth

Controls maximum depth.

```python
DecisionTreeClassifier(max_depth=5)
```

Smaller:

```text
Less complexity
Less overfitting
Possible underfitting
```

Larger:

```text
More complexity
Higher overfitting risk
```

---

## 2. min_samples_split

Minimum number of samples required to split an internal node.

Example:

```python
DecisionTreeClassifier(
    min_samples_split=10
)
```

A node containing fewer than the required samples will not be split.

---

## 3. min_samples_leaf

Minimum number of samples allowed in a leaf.

Example:

```python
DecisionTreeClassifier(
    min_samples_leaf=5
)
```

This prevents very small leaves.

---

## 4. max_leaf_nodes

Limits the maximum number of leaf nodes.

```python
DecisionTreeClassifier(
    max_leaf_nodes=20
)
```

---

## 5. criterion

For classification:

```python
criterion="gini"
```

or:

```python
criterion="entropy"
```

or:

```python
criterion="log_loss"
```

Current scikit-learn supports all three.

---

## 6. splitter

Two options:

```python
splitter="best"
```

or:

```python
splitter="random"
```

`best` searches for the best split, while `random` introduces randomness into candidate selection.

---

## 7. ccp_alpha

Controls cost-complexity pruning.

```python
DecisionTreeClassifier(
    ccp_alpha=0.01
)
```

---

# 📊 Feature Importance

Decision Trees can calculate feature importance.

Example:

```python
model.feature_importances_
```

Example output:

```text
Age            0.15
Income         0.55
CreditScore    0.30
```

Interpretation:

```text
Income
   ↓
Most influential among these features according to this tree's impurity-based importance
```

Scikit-learn exposes `feature_importances_` for the fitted Decision Tree.

### Important interview point

Feature importance does **not automatically mean causal importance**.

Also, impurity-based feature importance can be biased toward certain high-cardinality features.

For more robust interpretation, techniques such as:

```text
Permutation Importance
SHAP
```

can be considered.

---

# ✅ Advantages of Decision Trees

### 1. Easy to understand

Decision Trees are highly interpretable.

### 2. Easy to visualize

The tree can be plotted.

### 3. Handles nonlinear relationships

Example:

```text
If Income > X
AND Age < Y
```

### 4. Requires little feature scaling

Unlike algorithms such as:

```text
KNN
SVM
Logistic Regression
```

Decision Trees generally do not require standardization for the split mechanism.

### 5. Can perform classification and regression

```text
Classifier
Regressor
```

### 6. Captures feature interactions

Example:

```text
Age
+
Income
+
Credit Score
```

can interact through sequential splits.

---

# ❌ Disadvantages

### 1. Overfitting

Deep trees can memorize training data.

### 2. High variance

Small changes in training data can sometimes produce a substantially different tree.

### 3. Can become very large

A fully grown tree may become difficult to interpret.

### 4. Greedy learning

The tree generally chooses the best split locally at each step rather than globally optimizing the entire tree structure.

### 5. Feature importance can be misleading

Especially when using impurity-based importance with certain feature structures.

### 6. Single trees may be less accurate than ensemble methods

Random Forest and Gradient Boosting often provide stronger predictive performance.

---

# 🌲 Decision Tree vs Random Forest

A Random Forest is an ensemble of multiple Decision Trees.

## Decision Tree

```text
Dataset
   |
   v
One Tree
   |
   v
Prediction
```

## Random Forest

```text
Dataset
  |
  +----> Tree 1
  |
  +----> Tree 2
  |
  +----> Tree 3
  |
  +----> Tree 4
  |
  +----> Tree 5
          |
          v
    Combine Predictions
          |
          v
       Final Result
```

Random Forest uses many trees and combines their predictions, which generally reduces the variance associated with an individual tree.

---

# 🐍 Python — Decision Tree Classification

We can use the famous Iris dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# ------------------------------------------------
# 2. Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 3. Create Model
# ------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------

model.fit(X_train, y_train)

# ------------------------------------------------
# 5. Prediction
# ------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------
# 6. Evaluation
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
```

---

# 🌳 Decision Tree Visualization

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.show()
```

This gives a visual representation of:

```text
Feature
Threshold
Gini
Samples
Class
```

---

# 🔮 Prediction Example

Suppose we have:

```text
Sepal Length = 5.1
Sepal Width  = 3.5
Petal Length = 1.4
Petal Width  = 0.2
```

Python:

```python
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print(prediction)
```

Get class name:

```python
print(iris.target_names[prediction][0])
```

---

# 📊 Prediction Probability

Decision Trees can also return probabilities.

```python
probability = model.predict_proba(sample)

print(probability)
```

Example:

```text
[0.95, 0.05, 0.00]
```

This means the model estimates approximately:

```text
Class 0 → 95%
Class 1 → 5%
Class 2 → 0%
```

Scikit-learn calculates leaf probabilities from the fraction of training samples belonging to each class in that leaf.

---

# 📈 Decision Tree Regression

Decision Tree Regression predicts continuous values.

Example:

```text
Years Experience → Salary
```

Python:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Example dataset

data = pd.DataFrame({
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [30000, 35000, 40000, 45000,
               52000, 60000, 68000, 75000]
})

X = data[["Experience"]]
y = data["Salary"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create model

model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Train

model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)

# Metrics

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)
```

---

# 📐 Decision Tree Regression Concept

A Decision Tree Regressor divides the data into regions.

Example:

```text
Experience <= 3?
       /      \
     Yes       No
      |         |
 Experience   Experience
 <= 5?        <= 7?
```

Each leaf predicts a numerical value, commonly based on the training target values reaching that leaf.

---

# 📏 Model Evaluation

## Classification

Common metrics:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

Example:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

---

## Regression

Common metrics:

```text
MAE
MSE
RMSE
R²
```

Example:

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# 🔄 Complete Decision Tree ML Workflow

```text
                 Dataset
                    |
                    v
             Data Cleaning
                    |
                    v
             EDA / Analysis
                    |
                    v
             Feature Selection
                    |
                    v
             Train-Test Split
                    |
                    v
             Decision Tree
                    |
                    v
            Hyperparameter Tuning
                    |
                    v
                Pruning
                    |
                    v
              Model Evaluation
                    |
                    v
              Feature Importance
                    |
                    v
                Prediction
```

---

# 🎯 Hyperparameter Tuning

Example using GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("Best Score:")
print(grid.best_score_)
```

---

# ✂️ Finding a Good `ccp_alpha`

You can examine the cost-complexity pruning path:

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(X_train, y_train)

path = tree.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas

print(ccp_alphas)
```

Then test different alpha values:

```python
models = []

for alpha in ccp_alphas:

    model = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha
    )

    model.fit(X_train, y_train)

    models.append(model)
```

You can compare training and validation performance and select a suitable complexity level.

---

# 🎤 Important Interview Questions

## Beginner Level

### Q1. What is a Decision Tree?

**Answer:**

A Decision Tree is a supervised machine learning algorithm used for classification and regression. It recursively splits data based on feature conditions and produces predictions at leaf nodes.

---

### Q2. Is Decision Tree supervised or unsupervised?

**Answer:**

Decision Tree is a **supervised learning algorithm** because it learns from labeled target data.

---

### Q3. Can Decision Trees perform regression?

**Answer:**

Yes.

```python
DecisionTreeClassifier
```

is used for classification.

```python
DecisionTreeRegressor
```

is used for regression.

---

### Q4. What is a root node?

The root node is the first decision node of the tree.

---

### Q5. What is a leaf node?

A leaf node is the terminal node that contains the final prediction.

---

### Q6. What is a split?

A split divides the observations into child nodes based on a feature condition.

Example:

```text
Age <= 30
```

---

### Q7. What is tree depth?

Tree depth is the maximum number of edges/levels from the root to a leaf.

---

# 🧠 Intermediate Interview Questions

### Q8. What is Entropy?

Entropy measures uncertainty or impurity in a node.

```text
Entropy = -Σ p log2(p)
```

---

### Q9. What is Information Gain?

Information Gain measures the reduction in entropy after a split.

```text
IG = Parent Entropy - Weighted Child Entropy
```

---

### Q10. What is Gini Impurity?

Gini measures node impurity.

```text
Gini = 1 - Σp²
```

---

### Q11. What is the difference between Entropy and Gini?

Entropy uses logarithms while Gini does not.

Gini is generally computationally cheaper and is the default classification criterion in current scikit-learn.

---

### Q12. What is overfitting in Decision Trees?

Overfitting occurs when a tree becomes too complex and learns noise or very specific patterns from training data.

---

### Q13. How do you prevent overfitting?

Use:

```text
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
ccp_alpha
cross-validation
```

---

### Q14. What is pruning?

Pruning means removing unnecessary branches from a Decision Tree to reduce complexity and improve generalization.

---

### Q15. What is pre-pruning?

Pre-pruning limits tree growth while the tree is being built.

Examples:

```text
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
```

---

### Q16. What is post-pruning?

Post-pruning first builds the tree and then removes unnecessary branches.

Example:

```text
ccp_alpha
```

---

### Q17. What is CART?

CART stands for:

```text
Classification And Regression Trees
```

It supports classification and regression and uses binary splitting. Scikit-learn's tree implementation is based on an optimized CART approach.

---

### Q18. What is ID3?

ID3 is an early Decision Tree algorithm that uses:

```text
Entropy
Information Gain
```

---

### Q19. What is C4.5?

C4.5 is an extension of ID3 that uses:

```text
Gain Ratio
```

and adds improvements such as handling continuous features and pruning.

---

### Q20. What is Gain Ratio?

```text
Gain Ratio =
Information Gain / Split Information
```

It helps reduce the bias of Information Gain toward features with many possible values.

---

# 🔥 Advanced Interview Questions

### Q21. Why are Decision Trees prone to overfitting?

Because a tree can keep splitting until individual observations or very small groups are isolated.

Therefore:

```text
Very deep tree
      ↓
Low training error
      ↓
High variance
      ↓
Potential overfitting
```

---

### Q22. Why does increasing `max_depth` sometimes reduce test performance?

Because a larger depth allows the tree to learn increasingly detailed patterns, including noise.

---

### Q23. What happens if `max_depth=1`?

The tree can make only a very simple decision structure.

This may cause:

```text
Underfitting
```

---

### Q24. What happens if `max_depth=None`?

The tree can continue expanding until other stopping conditions are reached.

This can produce a very large tree and increase overfitting risk. Scikit-learn explicitly notes that fully grown, unpruned trees can become very large.

---

### Q25. Why don't Decision Trees usually require feature scaling?

Because Decision Trees compare feature values to thresholds.

For example:

```text
Age <= 30
```

Scaling changes the numerical representation but does not fundamentally change the ordering of observations.

---

### Q26. Can Decision Trees handle nonlinear relationships?

Yes.

This is one of their important strengths.

---

### Q27. What is feature importance?

Feature importance estimates how much a feature contributes to reducing impurity across the tree.

Scikit-learn exposes this through:

```python
model.feature_importances_
```



---

### Q28. Is feature importance the same as causality?

No.

A feature can be predictive without causing the target outcome.

---

### Q29. What is the difference between `min_samples_split` and `min_samples_leaf`?

`min_samples_split` controls how many samples are required before a node can be split.

```text
min_samples_split
```

controls splitting.

`min_samples_leaf` controls the minimum number of observations that must remain in each leaf.

```text
min_samples_leaf
```

controls leaf size.

---

### Q30. What is `ccp_alpha`?

`ccp_alpha` controls Minimal Cost-Complexity Pruning.

Higher alpha generally encourages a simpler tree.

---

### Q31. What is the difference between a Decision Tree and Random Forest?

Decision Tree:

```text
One tree
```

Random Forest:

```text
Many trees
+
Randomness
+
Aggregation
```

Random Forest generally reduces the variance of individual trees.

---

### Q32. Why can Random Forest perform better than a single Decision Tree?

A single Decision Tree can have high variance.

Random Forest averages predictions across many decorrelated trees, reducing variance and often improving generalization.

---

### Q33. What is high variance in a Decision Tree?

High variance means the model's predictions can change significantly when the training data changes slightly.

Decision Trees are often considered high-variance learners.

---

### Q34. What is bias in a Decision Tree?

Bias represents error caused by overly simplistic assumptions.

A very shallow tree can have:

```text
High Bias
Low Variance
```

A very deep tree can have:

```text
Low Training Bias
High Variance
```

---

### Q35. What is the bias-variance tradeoff?

```text
Very Simple Tree
       ↓
High Bias
Low Variance

Very Complex Tree
       ↓
Low Bias
High Variance
```

The goal is to find a model complexity that generalizes well.

---

# 💼 Practical Data Science Interview Questions

### Q36. Your Decision Tree has 100% training accuracy and 70% test accuracy. What would you do?

I would investigate overfitting.

I could:

```text
Reduce max_depth
Increase min_samples_leaf
Increase min_samples_split
Limit max_leaf_nodes
Use pruning
Use cross-validation
Tune hyperparameters
Compare with Random Forest
```

---

### Q37. Your Decision Tree is underfitting. What would you do?

Possible actions:

```text
Increase max_depth
Decrease min_samples_leaf
Decrease min_samples_split
Allow more leaf nodes
Review feature quality
Review data preprocessing
```

---

### Q38. Would you use StandardScaler before a Decision Tree?

Usually, no.

Decision Trees generally do not require feature scaling because they select splits based on feature thresholds.

---

### Q39. How would you tune a Decision Tree?

I would use:

```text
GridSearchCV
RandomizedSearchCV
Cross-validation
```

Important parameters:

```text
criterion
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
ccp_alpha
```

---

### Q40. How would you explain a Decision Tree to a non-technical stakeholder?

I would say:

> "It works like a flowchart. At each step, the model asks a question about the data. Depending on the answer, it follows one branch until it reaches a final decision."

---

# 🏦 Banking Interview Example

Suppose we are building a loan approval model.

Features:

```text
Age
Income
Credit Score
Existing Loans
Employment Length
Debt-to-Income Ratio
```

Target:

```text
Loan Approved
```

The tree might learn:

```text
              Credit Score > 700?
                 /          \
               Yes           No
                |             |
        Income > 50K?       Reject
          /       \
        Yes       No
         |         |
      Approve    Reject
```

The advantage is interpretability.

A business user can understand:

```text
Credit Score > 700
AND
Income > 50K
→ Higher likelihood of approval
```

However, in real banking applications, model governance, fairness, stability, explainability, validation, monitoring, and regulatory requirements must also be considered.

---

# 🧪 Complete Practical Example

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. Create Example Dataset
# ==========================================

data = pd.DataFrame({
    "Age": [22, 25, 28, 35, 40, 45, 50, 52, 30, 33],
    "Income": [
        25000, 28000, 30000, 55000, 65000,
        80000, 90000, 95000, 45000, 50000
    ],
    "CreditScore": [
        580, 600, 620, 700, 720,
        750, 780, 800, 680, 690
    ],
    "Approved": [
        0, 0, 0, 1, 1,
        1, 1, 1, 0, 1
    ]
})

# ==========================================
# 2. Features and Target
# ==========================================

X = data[
    [
        "Age",
        "Income",
        "CreditScore"
    ]
]

y = data["Approved"]

# ==========================================
# 3. Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# 4. Create Decision Tree
# ==========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# ==========================================
# 5. Train
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 6. Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. Evaluation
# ==========================================

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 8. Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(
    by="Importance",
    ascending=False
))

# ==========================================
# 9. Visualize Tree
# ==========================================

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True
)

plt.show()
```

---

# 🧠 Decision Tree Mental Model

Remember this sequence:

```text
DATA
 ↓
ROOT
 ↓
BEST FEATURE
 ↓
BEST THRESHOLD
 ↓
SPLIT
 ↓
CHILD NODES
 ↓
REPEAT
 ↓
LEAF
 ↓
PREDICTION
```

For classification:

```text
Best split
=
lowest impurity
```

or, depending on the algorithm:

```text
highest information gain
```

For regression:

```text
Best split
=
largest reduction in prediction error/variance
```

---

# 📝 One-Minute Interview Answer

If an interviewer asks:

### "Explain Decision Tree."

A strong answer:

> "Decision Tree is a supervised machine learning algorithm used for both classification and regression. It works like a flowchart where each internal node represents a feature-based decision, each branch represents an outcome of that decision, and each leaf represents the final prediction. During training, the algorithm recursively selects splits that improve the chosen criterion, such as Gini impurity or entropy for classification and error/variance reduction for regression. Decision Trees are easy to interpret and generally don't require feature scaling, but they can overfit easily. I control this using parameters such as max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, and pruning techniques such as cost-complexity pruning."

---

# 🚀 Quick Revision

Remember these points for interviews:

```text
Decision Tree
      ↓
Supervised Learning
      ↓
Classification + Regression
      ↓
Root
      ↓
Internal Nodes
      ↓
Branches
      ↓
Leaves
```

### Classification

```text
Gini
Entropy
Information Gain
Gain Ratio
```

### Regression

```text
MSE
Variance Reduction
```

### Algorithms

```text
ID3
C4.5
CART
```

### Overfitting Control

```text
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
ccp_alpha
```

### Evaluation

```text
Classification:
Accuracy
Precision
Recall
F1
ROC-AUC

Regression:
MAE
MSE
RMSE
R²
```

### Important Relationship

```text
Deep Tree
   ↓
More Complex
   ↓
Lower Training Error
   ↓
Higher Overfitting Risk
```

```text
Pruning
   ↓
Smaller Tree
   ↓
Lower Complexity
   ↓
Better Generalization
```

---

# 📚 Key Interview Topics Checklist

Before attending a Machine Learning interview, make sure you can explain:

- [x] What is Decision Tree?
- [x] Classification vs Regression
- [x] Root Node
- [x] Internal Node
- [x] Leaf Node
- [x] Branch
- [x] Depth
- [x] Split
- [x] Impurity
- [x] Entropy
- [x] Information Gain
- [x] Gini Impurity
- [x] Gain Ratio
- [x] Variance Reduction
- [x] ID3
- [x] C4.5
- [x] CART
- [x] Overfitting
- [x] Underfitting
- [x] Pre-Pruning
- [x] Post-Pruning
- [x] Cost-Complexity Pruning
- [x] `max_depth`
- [x] `min_samples_split`
- [x] `min_samples_leaf`
- [x] `max_leaf_nodes`
- [x] `ccp_alpha`
- [x] Feature Importance
- [x] Decision Tree vs Random Forest
- [x] Decision Tree vs Logistic Regression
- [x] Decision Tree vs KNN
- [x] Decision Tree vs SVM
- [x] Hyperparameter tuning
- [x] Cross-validation
- [x] Classification metrics
- [x] Regression metrics
- [x] Python implementation
- [x] Tree visualization

---

# 🔗 References

1. Scikit-learn — DecisionTreeClassifier  
   https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

2. Scikit-learn — Decision Trees User Guide  
   https://scikit-learn.org/stable/modules/tree.html

3. Breiman et al. — Classification and Regression Trees (CART)

4. Quinlan — ID3 / C4.5 Decision Tree Algorithms

---

# 👨‍💻 Author

**Machine Learning Study Repository**

Topics covered:

```text
Python
Pandas
NumPy
Scikit-learn
Machine Learning
Decision Trees
Classification
Regression
Model Evaluation
Hyperparameter Tuning
```

---

# ⭐ Final Concept

The easiest way to remember a Decision Tree is:

```text
              QUESTION
                 |
          ┌──────┴──────┐
         YES            NO
          |              |
      QUESTION        QUESTION
       /   \            /   \
      /     \          /     \
    YES      NO       YES      NO
     |        |        |        |
   LEAF     LEAF      LEAF     LEAF
```

A Decision Tree is essentially a **machine-learned flowchart**.

It learns:

```text
IF condition
THEN go left

ELSE
go right

UNTIL
leaf node

THEN
make prediction
```

That is the core idea behind Decision Trees.