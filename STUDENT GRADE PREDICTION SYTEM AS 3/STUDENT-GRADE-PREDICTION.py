import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("cgpaPRED.csv")

df.shape
df.describe()
df.info()
df.columns

df.duplicated().sum()
df = df.drop_duplicates()
df.shape
df

df.notna().sum()
df.isnull().sum()

X = df[['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']]
y = df['SEM 5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_simple = X_train[['SEM 1']]
X_test_simple = X_test[['SEM 1']]

a = df["SEM 1"]
b = df["SEM 5"]

plt.scatter(a, b, color="blue", label='Data Points')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()

semesterCols = ["SEM 1", "SEM 2", "SEM 3", "SEM 4", "SEM 5"]
dataToPolot = [df[col].dropna() for col in semesterCols]

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(dataToPolot, tick_labels=semesterCols, patch_artist=True)
ax.set_title("Box plot of Students performance")
ax.set_xlabel("Semesters")
ax.set_ylabel("Scores")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

corrMatrix = df[semesterCols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corrMatrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Semester Scores")
plt.show()

corrWithLabel = corrMatrix["SEM 5"].sort_values(ascending=False)
print("Correlation of each feature with SEM 5 (label):")
print(corrWithLabel)

model = LinearRegression()
model.fit(X_train_simple, y_train)

y_pred = model.predict(X_test_simple)

print("\nRegression Equation:")
print(f"Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X1")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0

print("MODEL PERFORMANCE")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual SEM 5")
plt.ylabel("Predicted SEM 5")
plt.title("Simple Linear Regression: Actual vs Predicted")
plt.show()

model = LinearRegression()
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)

print("Regression Equation:")
print(
    f"Y = {model.intercept_:.2f} + "
    f"{model.coef_[0]:.2f}X1 + "
    f"{model.coef_[1]:.2f}X2 + "
    f"{model.coef_[2]:.2f}X3 + "
    f"{model.coef_[3]:.2f}X4"
)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0

print("MODEL PERFORMANCE") 
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual SEM 5")
plt.ylabel("Predicted SEM 5")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.show()

df['SEM5_check'] = df[['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']].mean(axis=1)
comparison = df[['SEM 5', 'SEM5_check']].copy()
comparison['difference'] = comparison['SEM 5'] - comparison['SEM5_check']
print(comparison.describe()) 