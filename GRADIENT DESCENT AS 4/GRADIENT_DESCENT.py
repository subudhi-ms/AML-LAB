import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3], dtype=float)
y = np.array([2,3,4], dtype=float)

n = len(x)
print("Input x : ", x)
print("Actual y : ", y)
print("Number of samples : ", n)

w = 0.0
b = 0.0
alpha = 0.1
iterations = 2
print("Initial weight (w) : ", w)
print("Initial bias (b) : ", b)
print("Learning rate : ", alpha)

def predict(x,w,b):
  return w*x+b

y_pred = predict(x,w,b)
print("Predictions : ", y_pred)

def compute_cost(y, y_pred):
  return np.sum((y-y_pred)**2)/n

y_pred = predict(x,w,b)
cost = compute_cost(y,y_pred)
print("\nInitial prediction", y_pred)
print("Initial Cost : ", cost)

def gradient(x,y,y_pred):
  dw = (np.sum(x*(y_pred-y))*2)/n
  db = (np.sum(y_pred-y)*2)/n
  return dw, db

def parameter_update(w,b,dw,db,alpha):
  w = w - alpha*dw
  b = b - alpha*db
  return w,b

costs = []
w_history = []
b_history = []


for i in range(iterations):
  y_pred_current_params = predict(x,w,b)
  dw, db = gradient(x,y,y_pred_current_params)
  w, b = parameter_update(w,b,dw,db,alpha)
  y_pred_after_update = predict(x,w,b)
  cost_after_update = compute_cost(y,y_pred_after_update)

  costs.append(cost_after_update)
  w_history.append(w)
  b_history.append(b)
  print("-"*30)
  print(f"\nIteration {i+1}")
  print("The y_predict (after update) is ", y_pred_after_update)
  print("The cost (after update) is ", cost_after_update)
  print(f"Gradients dw: {dw}, db: {db}")
  print("The updated weight (w) is ", w)
  print("The updated bias (b) is ", b)

final_w = w_history[-1]
final_b = b_history[-1]
print("\nThe final equation is : ")
print(f"y = {final_w}x + {final_b}")
print("The final cost is ", costs[-1])

plt.plot(range(iterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function VS Iterations')
plt.show()


plt.scatter(x,y, label='Actual Data')
if iterations > 0:
    final_y_pred_for_plot = predict(x, final_w, final_b)
else:
    final_y_pred_for_plot = predict(x, w, b)
plt.plot(x, final_y_pred_for_plot, color='black', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()