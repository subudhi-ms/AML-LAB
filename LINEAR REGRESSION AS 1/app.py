import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Calculate Linear Regression
@app.route("/calculate", methods=["POST"])
def calculate():

    data = request.get_json()

    x = data["x"]
    y = data["y"]

    n = len(x)

    if n < 2:
        return jsonify({"error": "Enter at least 2 points"})

    # Mean
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Slope
    numerator = 0
    denominator = 0

    for i in range(n):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2

    if denominator == 0:
        return jsonify({"error": "Invalid X values"})

    m = numerator / denominator
    c = mean_y - m * mean_x

    # Predicted values
    y_pred = []

    for value in x:
        y_pred.append(m * value + c)

    # Error calculations
    mae = 0
    mse = 0

    for i in range(n):
        mae += abs(y[i] - y_pred[i])
        mse += (y[i] - y_pred[i]) ** 2

    mae = mae / n
    mse = mse / n
    rmse = mse ** 0.5

    ss_res = 0
    ss_tot = 0

    for i in range(n):
        ss_res += (y[i] - y_pred[i]) ** 2
        ss_tot += (y[i] - mean_y) ** 2

    if ss_tot == 0:
        r2 = 0
    else:
        r2 = 1 - (ss_res / ss_tot)

    # Plot
    plt.figure(figsize=(6,4))

    plt.scatter(x, y)

    plt.plot(x, y_pred)

    plt.xlabel("X")

    plt.ylabel("Y")

    image = io.BytesIO()

    plt.savefig(image, format="png")

    image.seek(0)

    graph = base64.b64encode(image.getvalue()).decode()

    plt.close()

    return jsonify({

        "mean_x": round(mean_x,2),

        "mean_y": round(mean_y,2),

        "slope": round(m,2),

        "intercept": round(c,2),

        "equation": f"Y = {round(m,2)}X + {round(c,2)}",

        "mae": round(mae,2),

        "mse": round(mse,2),

        "rmse": round(rmse,2),

        "r2": round(r2,2),

        "plot_url": graph

    })


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    x = float(data["x"])

    m = float(data["m"])

    c = float(data["c"])

    y = m * x + c

    return jsonify({"prediction": round(y,2)})


if __name__ == "__main__":
    app.run(debug=True)