import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  
n = int(input("Enter the number of students: "))

study_hours = []
attendance = []
assignment = []
final_marks = []

print("\nEnter the details:\n")

for i in range(n):
    print(f"Student {i+1}")
    study_hours.append(float(input("Study Hours: ")))
    attendance.append(float(input("Attendance (%): ")))
    assignment.append(float(input("Assignment Marks: ")))
    final_marks.append(float(input("Final Marks: ")))
    print()

data = pd.DataFrame({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Assignment": assignment,
    "Final Marks": final_marks
})

print("\nDataset")
print(data)

X = data[["Study Hours", "Attendance", "Assignment"]]
y = data["Final Marks"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\nRegression Equation:\n")

print(f"Final Marks = {model.intercept_:.2f}"
      f" + ({model.coef_[0]:.2f} × Study Hours)"
      f" + ({model.coef_[1]:.2f} × Attendance)"
      f" + ({model.coef_[2]:.2f} × Assignment)")

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\nError Measures")
print("---------------------")
print("Mean Absolute Error (MAE):", round(mae,3))
print("Mean Squared Error (MSE):", round(mse,3))
print("Root Mean Squared Error (RMSE):", round(rmse,3))
print("R2 Score:", round(r2,3))

result = pd.DataFrame({
    "Actual Marks": y,
    "Predicted Marks": y_pred
})

print("\nActual vs Predicted")
print(result)

print("\nEnter new student's details for prediction")

new_hours = float(input("Study Hours: "))
new_attendance = float(input("Attendance (%): "))
new_assignment = float(input("Assignment Marks: "))

new_data = [[new_hours, new_attendance, new_assignment]]

prediction = model.predict(new_data)

print("\nPredicted Final Marks =", round(prediction[0],2))

plt.figure(figsize=(6,5))
plt.scatter(y, y_pred, color='blue', s=70)

min_val = min(min(y), min(y_pred))
max_val = max(max(y), max(y_pred))

plt.plot([min_val, max_val],
         [min_val, max_val],
         color='red',
         linewidth=2)

plt.title("Actual vs Predicted Marks")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.grid(True)

plt.figure(figsize=(6,5))

residuals = y - y_pred

plt.scatter(y_pred, residuals, color='green', s=70)
plt.axhline(y=0, color='red', linestyle='--')

plt.title("Residual Plot")
plt.xlabel("Predicted Marks")
plt.ylabel("Residual Error")
plt.grid(True)

plt.show()