import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# --------------------------------------------------------
# 1. Setup MLflow Tracking
# --------------------------------------------------------
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint

mlflow.set_tracking_uri("http://34.70.222.183:8100")
client = MlflowClient(mlflow.get_tracking_uri())
all_experiments = client.search_experiments()
print(all_experiments)

mlflow.set_experiment("Iris-Classifier-MLflow-Demoooooo")
# --------------------------------------------------------
# 2. Load Data
# --------------------------------------------------------
data = pd.read_csv("data/iris.csv")

train, test = train_test_split(data, test_size=0.4, stratify=data["species"], random_state=42)
X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train.species
X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_test = test.species

# --------------------------------------------------------
# 3. Define Hyperparameter Search Space
# --------------------------------------------------------
space = {
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
}

# --------------------------------------------------------
# 4. Define Objective Function for Hyperopt
# --------------------------------------------------------
def objective(params):
    params["max_depth"] = int(params["max_depth"])
    params["min_samples_split"] = int(params["min_samples_split"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {"loss": 1 - acc, "status": STATUS_OK}


# --------------------------------------------------------
# 5. Hyperparameter Optimization
# --------------------------------------------------------
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(42))

# Decode categorical params
criterion_list = ["gini", "entropy"]
best_params["criterion"] = criterion_list[best_params["criterion"]]
best_params["max_depth"] = int(best_params["max_depth"])
best_params["min_samples_split"] = int(best_params["min_samples_split"])
best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

print("\nBest Hyperparameters found:")
print(best_params)

# --------------------------------------------------------
# 6. Train Final Model with Best Hyperparameters
# --------------------------------------------------------
final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nFinal model accuracy: {final_acc:.4f}")

# --------------------------------------------------------
# 7. Log Final Model in MLflow and Register
# --------------------------------------------------------
with mlflow.start_run(run_name="Best Model Training"):
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", final_acc)

    # Model signature (optional but recommended)
    signature = infer_signature(X_train, final_model.predict(X_train))

    # Log and register model
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="iris_model",
        registered_model_name="Iris-DT-Classifier",
        signature=signature,
        input_example=X_train.head()
    )

# --------------------------------------------------------
# 8. Save Model and Metrics Locally (for backward compatibility)
# --------------------------------------------------------
joblib.dump(final_model, "model.joblib")
with open("metrics.csv", "w") as f:
    f.write("accuracy,%.4f\n" % final_acc)

# --------------------------------------------------------
# 9. (Optional) Automatically promote the latest model to Production
# --------------------------------------------------------
try:
    client = MlflowClient()
    latest_version = client.get_latest_versions("Iris-DT-Classifier", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="Iris-DT-Classifier",
        version=latest_version,
        stage="Production"
    )
    print(f"✅ Model version {latest_version} promoted to Production.")
except Exception as e:
    print("⚠️ Could not promote model automatically:", e)
