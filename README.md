# Multi-Feature Linear Regression

A simple implementation of **Multi-Feature Linear Regression from scratch using Python, NumPy, and Pandas**.

This project explores how linear regression works internally by implementing feature normalization, cost calculation, and gradient descent without relying on machine learning libraries such as Scikit-learn.

The model is trained to predict the number of goals scored by football players using several performance statistics.

## Overview

Unlike simple linear regression, which uses a single feature, multi-feature linear regression uses multiple input variables to make predictions.

In this project, the model uses three features:

- **xG (Expected Goals):** The expected number of goals based on the quality of chances.
- **Shots:** Total shots taken by a player.
- **OnTarget:** Number of shots on target.

The target variable is **Goals**, representing the actual number of goals scored.

## How It Works

The project implements the main steps of a linear regression model:

### 1. Feature Normalization

Features are normalized using Z-score normalization (Standardization):

\[
x_{normalized} = \frac{x-\mu}{\sigma}
\]

Where:
- \(x\) is the original feature value.
- \(\mu\) is the feature mean.
- \(\sigma\) is the feature standard deviation.

This helps keep features on comparable scales and makes gradient descent more stable.

### 2. Target Normalization

The target variable is also standardized before training. Predictions are converted back to the original scale afterward.

### 3. Cost Function

The model uses the Mean Squared Error-based cost function:

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}
(f_{w,b}(x^{(i)})-y^{(i)})^2
\]

Where:

\[
f_{w,b}(x)=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

### 4. Gradient Descent

Gradient descent iteratively updates the weights and bias to minimize the cost function.

The parameters are updated using:

\[
w_j := w_j-\alpha\frac{\partial J}{\partial w_j}
\]

\[
b := b-\alpha\frac{\partial J}{\partial b}
\]

Where \(\alpha\) represents the learning rate.

## Project Structure

```text
multi-feature-linear-regression/
│
├── example data/
│   └── Data.csv
│
├── src/
│   ├── example.py
│   │
│   └── multi_feature_linear_regression/
│       └── __init__.py
│
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd multi-feature-linear-regression
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib
```

If you're using `uv`, you can install the dependencies with:

```bash
uv add numpy pandas matplotlib
```

## Running the Project

Run the example script from the project root:

```bash
python src/example.py
```

Or, if you're using `uv`:

```bash
uv run python src/example.py
```

The script will:

1. Load the football players dataset.
2. Select the input features and target variable.
3. Normalize the data.
4. Train the model using gradient descent.
5. Display the training progress, including the cost, weights, and bias.
6. Plot actual goals against predicted goals for the first 51 players.

## Technologies Used

- **Python** – Main programming language.
- **NumPy** – Matrix operations and numerical calculations.
- **Pandas** – Data loading, manipulation, and normalization.
- **Matplotlib** – Visualizing actual and predicted values.

## Learning Objectives

This project was built as a practical exercise to understand the mathematics and implementation behind linear regression, particularly:

- Multi-feature linear regression.
- Feature and target normalization.
- Cost functions.
- Gradient descent.
- Vectorization using NumPy.
- Data visualization using Matplotlib.

---

**Note:** This is an educational implementation intended to explore the fundamentals of machine learning algorithms rather than a production-ready prediction system.
