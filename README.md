# Multi-Feature Linear Regression

A simple implementation of **multi-feature linear regression** from scratch using Python, NumPy, and Pandas.

The project uses gradient descent to predict football players' goals based on their performance statistics.

## Overview

The model uses the following features to predict the number of goals scored by a player:

- **xG** – Expected Goals
- **Shots** – Total shots
- **OnTarget** – Shots on target

The target variable is **Goals**.

Before training, numerical features and the target values are standardized using Z-score normalization.

## Results

<img width="1799" height="879" alt="Figure_1" src="https://github.com/user-attachments/assets/1bdf56ff-1c5c-4f4e-8598-4c1481d194eb" />

The plot compares the actual goals with the model's predictions
for the first 51 players

## How It Works

### 1. Feature Normalization

Numerical features are standardized so that they have a mean of 0 and a standard deviation of 1.

The normalization formula is:

$$
x_{\text{normalized}} = \frac{x - \mu}{\sigma}
$$

Where:

- $x$ is the original value.
- $\mu$ is the mean of the feature.
- $\sigma$ is the standard deviation of the feature.

Non-numerical columns are left unchanged.

### 2. Multi-Feature Linear Regression

The model predicts the target using a weighted sum of the input features plus a bias.

$$
f(x) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Where:

- $w_j$ is the weight of feature $j$.
- $x_j$ is the value of feature $j$.
- $b$ is the bias.

### 3. Cost Function

The model uses Mean Squared Error with a factor of one half to measure prediction error:

$$
J(w,b) = \frac{1}{2m}
\sum_{i=1}^{m}
\left(f(x^{(i)}) - y^{(i)}\right)^2
$$

Where:

- $m$ is the number of training examples.
- $f(x^{(i)})$ is the predicted value.
- $y^{(i)}$ is the actual value.

### 4. Gradient Descent

Gradient descent updates the weights and bias iteratively to minimize the cost function.

The update rules are:

$$
w_j := w_j - \alpha
\frac{\partial J(w,b)}{\partial w_j}
$$

$$
b := b - \alpha
\frac{\partial J(w,b)}{\partial b}
$$

Where $\alpha$ is the learning rate.

The implementation uses NumPy operations to calculate predictions, errors, and gradients efficiently.

## Project Structure

```text
multi-feature-linear-regression/
│
├── example data/
│   └── Data.csv
│
├── src/
│   ├── example.py
│   └── multi_feature_linear_regression/
│       └── __init__.py
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python
- [uv](https://docs.astral.sh/uv/)

### 1. Clone the repository

```bash
git clone <repository-url>
cd multi-feature-linear-regression
```

### 2. Install dependencies

Run:

```bash
uv sync
```

This creates the virtual environment (if needed) and installs the dependencies defined in `pyproject.toml`.

### 3. Run the example

```bash
uv run python src/example.py
```

The script trains the model and displays a plot comparing the actual and predicted goals for the first 51 players.

## Technologies Used

- **Python** – Main programming language
- **NumPy** – Numerical computations and vectorized gradient descent
- **Pandas** – Data loading and manipulation
- **Matplotlib** – Visualizing actual and predicted values
- **uv** – Python project and dependency management

## Learning Objectives

This project was built to practice implementing machine learning concepts from scratch, including:

- Multi-feature linear regression
- Z-score normalization
- Cost function calculation
- Gradient descent
- Vectorized operations with NumPy
- Comparing predictions with actual data

## Note

This is an educational implementation intended to help understand the mathematics and mechanics behind linear regression. It is not intended as a production-ready prediction model.
