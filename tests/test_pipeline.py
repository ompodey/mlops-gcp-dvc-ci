import pandas as pd
import joblib
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import mlflow
import mlflow.sklearn


MLFLOW_TRACKING_URI = "http://34.60.145.174:8100"
MODEL_NAME = "Iris-DT-Classifier"

def load_and_fix_metrics(file_path="metrics.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) >= 2:
        df = pd.read_csv(file_path)
        return df

    elif len(lines) == 1 and "," in lines[0]:
        header, value = lines[0].split(",")
        df = pd.DataFrame({header: [float(value)]})
        df.to_csv(file_path, index=False)
        print(f"Fixed the file, {file_path} format corrected.")
        return df

    else:
        raise ValueError(f"Unexpected format in {file_path}. Content:\n{lines}")


@pytest.fixture
def data():
    df = pd.read_csv("data/iris.csv")
    return df

@pytest.fixture
def model():
    """Fetch the latest registered model from MLflow Model Registry"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["Production", "None"])
    if not latest:
        raise Exception(f"No model found in MLflow registry for {MODEL_NAME}")

    model_uri = f"models:/{MODEL_NAME}/{latest[0].version}"
    print(f"Fetching model from MLflow registry: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    return model

#Sanity
def test_data_sanity(data):
    assert data.isnull().sum().sum() ==0, "Data contains null values!"
    
    expected_cols =['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in data.columns for col in expected_cols), "Missing expected columns!"
    print(f"\nSanity check sucessfullly passed : {len(data)} rows, columns: {list(data.columns)}")


def test_model_evaluation(data, model):
    _, eval_data = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=42)
    X_eval = eval_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_eval =eval_data['species']
    y_pred= model.predict(X_eval)
    test_acc = accuracy_score(y_eval, y_pred)

    sample_df = eval_data.copy()
    sample_df['predicted'] = y_pred
    sample_df['test_accuracy'] = test_acc
    sample_df = sample_df.head(20) 
    sample_df.to_csv("metrics_test.csv", index=False)

    print("\n Evaluation Metrics Sample:")
    print(sample_df.to_string(index=False))
    
  
    print(f" Test Accuracy (current run): {test_acc}")

    assert test_acc > 0.7, "Test accuracy too low."
