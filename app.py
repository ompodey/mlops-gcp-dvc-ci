# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load the model (path inside the container will be /app/model/model.joblib)
MODEL_PATH = "model/model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("Failed to load model:", e)
    traceback.print_exc()

@app.route("/", methods=["GET"])
def home():
    return {"message": "Iris Model API is up!"}

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON body received"}), 400

    # Accept either a single dict or a list of dicts
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400

    # Expected feature order
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    try:
        X = df[feature_cols]
    except KeyError as e:
        return jsonify({"error": f"Missing columns. Expected: {feature_cols}"}), 400

    preds = model.predict(X)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
