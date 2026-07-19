# model.py

# Take 4 X values from user
x = []
print("Enter 4 X values:")
for i in range(4):
    value = float(input(f"X{i+1}: "))
    x.append(value)

# Take 4 Y values from user
y = []
print("\nEnter 4 Y values:")
for i in range(4):
    value = float(input(f"Y{i+1}: "))
    y.append(value)

n = len(x)

# Mean
mean_x = sum(x) / n
mean_y = sum(y) / n

# Calculate slope (m)
num = 0
den = 0

for i in range(n):
    num += (x[i] - mean_x) * (y[i] - mean_y)
    den += (x[i] - mean_x) ** 2

m = num / den

# Calculate intercept (c)
c = mean_y - (m * mean_x)

# Prediction
y_pred = []

for value in x:
    y_pred.append(m * value + c)

# MAE
mae = 0
for i in range(n):
    mae += abs(y[i] - y_pred[i])

mae = mae / n

# MSE
mse = 0
for i in range(n):
    mse += (y[i] - y_pred[i]) ** 2

mse = mse / n

# RMSE
rmse = mse ** 0.5

# R² Score
ss_res = 0
ss_tot = 0

for i in range(n):
    ss_res += (y[i] - y_pred[i]) ** 2
    ss_tot += (y[i] - mean_y) ** 2

if ss_tot == 0:
    r2 = 0
else:
    r2 = 1 - (ss_res / ss_tot)

# Display Results
print("\nLinear Regression Model")
print("-----------------------")    
print("Mean X :", round(mean_x, 2))
print("Mean Y :", round(mean_y, 2))
print("Slope (m) :", round(m, 4))
print("Intercept (c) :", round(c, 4))
print("Equation : Y =", round(m, 4), "X +", round(c, 4))
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R² Score :", round(r2, 4))

# Predict New Value
new_x = float(input("\nEnter X value to predict Y: "))
new_y = m * new_x + c

print("Predicted Y =", round(new_y, 4))