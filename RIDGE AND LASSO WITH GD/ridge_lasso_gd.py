# RIDGE GD
import numpy as np
import matplotlib.pyplot as plt

def ridge_regression_gd(X, y, learning_rate, n_iterations, lambda_param):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    cost_history = []
    for i in range(n_iterations):
        y_predicted = np.dot(X, weights) + bias
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (2 * lambda_param * weights)
        db = (1 / n_samples) * np.sum(y_predicted - y)
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2) + (lambda_param * np.sum(weights**2))
        cost_history.append(cost)

    return weights, bias, cost_history

x_input = input("Enter the x feature : ")
y_input = input("Enter the y feature : ")

X = np.array([float(val) for val in x_input.split(',')]).reshape(-1, 1)
y = np.array([float(val) for val in y_input.split(',')])

print(X.flatten())

learning_rate = 0.01
n_iterations = 1000
lambda_param = 0.1

weights, bias, cost_history = ridge_regression_gd(X, y, learning_rate, n_iterations, lambda_param)

print(f"Weights: {weights}")
print(f"Bias: {bias}")

fig_cost = plt.figure(figsize=(6, 6))
plt.plot(range(n_iterations), cost_history)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost (MSE + L2 Regularization)")
plt.title("Cost History during Gradient Descent")
plt.grid(True)
plt.show()

fig_reg = plt.figure(figsize=(6, 6))
plt.scatter(X, y, label="Original Data")
plt.plot(X, np.dot(X, weights) + bias, color='blue', label="Ridge Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge Regression with Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# LASSO GD


import numpy as np
import matplotlib.pyplot as plt

def lasso_regression_gd(X, y, learning_rate, n_iterations, lambda_param):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    cost_history = []
    for i in range(n_iterations):
        y_predicted = np.dot(X, weights) + bias

        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + lambda_param * np.sign(weights)
        db = (1 / n_samples) * np.sum(y_predicted - y)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2) + (lambda_param * np.sum(np.abs(weights)))
        cost_history.append(cost)

    return weights, bias, cost_history

x_lasso_input = input("Enter the x feature for Lasso : ")
y_lasso_input = input("Enter the y feature for Lasso : ")

X_lasso = np.array([float(val) for val in x_lasso_input.split(',')]).reshape(-1, 1)
y_lasso = np.array([float(val) for val in y_lasso_input.split(',')])

learning_rate_lasso = 0.01
n_iterations_lasso = 1000
lambda_param_lasso = 0.1

weights_lasso, bias_lasso, cost_history_lasso = lasso_regression_gd(X_lasso, y_lasso, learning_rate_lasso, n_iterations_lasso, lambda_param_lasso)

print(f"Lasso Weights: {weights_lasso}")
print(f"Lasso Bias: {bias_lasso}")

fig_cost_lasso = plt.figure(figsize=(6, 6))
plt.plot(range(n_iterations_lasso), cost_history_lasso)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost (MSE + L1 Regularization)")
plt.title("Lasso Regression Cost History during Gradient Descent")
plt.grid(True)
plt.show()


fig_reg_lasso = plt.figure(figsize=(6, 6))
plt.scatter(X_lasso, y_lasso, label="Original Data")
plt.plot(X_lasso, np.dot(X_lasso, weights_lasso) + bias_lasso, color='orange', label="Lasso Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Lasso Regression with Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()