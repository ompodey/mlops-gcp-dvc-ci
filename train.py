import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pprint import pprint

# 1. Mlflow setup
mlflow.set_tracking_uri("http://34.60.145.174:8100")
clnt=MlflowClient()
print("Available Experiments:")
for exp in clnt.search_experiments():
    print(f" - {exp.name}")

mlflow.set_experiment("Mlfow-Screencast-Demo-Iris")

iris_df =pd.read_csv("data/iris.csv")

train_df,test_df=train_test_split(iris_df, test_size=0.4, stratify=iris_df["species"], random_state=42)

feature_cols= ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train, y_train =train_df[feature_cols], train_df["species"]
X_test, y_test = test_df[feature_cols], test_df["species"]
param_space = {
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1)
}

def evaluate_model(params_dict):
    # float to int conversion for specific hyperparameters
    params_dict={k: int(v) if isinstance(v, float) else v for k, v in params_dict.items()}

    with mlflow.start_run(nested=True):
        mlflow.log_params(params_dict)
        tree=DecisionTreeClassifier(**params_dict, random_state=42)
        tree.fit(X_train, y_train)

        y_pred=tree.predict(X_test)
        acc_val= accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc_val)
        mlflow.sklearn.log_model(tree, "model")

        return {"loss":1-acc_val, "status": STATUS_OK}

trial_records = Trials()
best_combo = fmin(fn=evaluate_model,space=param_space,algo=tpe.suggest,max_evals=10,trials=trial_records,rstate=np.random.default_rng(42))


criteria_options = ["gini", "entropy"]
best_combo["criterion"] =criteria_options[best_combo["criterion"]]


for param in ["max_depth", "min_samples_split"]:
    best_combo[param] = int(best_combo[param])

print("\nOptimal Parameters Identified:")
print(best_combo)


final_tree=DecisionTreeClassifier(**best_combo, random_state=42)
final_tree.fit(X_train, y_train)

predicted_labels = final_tree.predict(X_test)
final_accuracy=accuracy_score(y_test, predicted_labels)
print(f"\nFinal Accuracy Achieved: {final_accuracy:.4f}")


with mlflow.start_run(run_name="Best Model Training"):
    mlflow.log_params(best_combo)
    mlflow.log_metric("accuracy", final_accuracy)

    model_signature=infer_signature(X_train, final_tree.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=final_tree,
        artifact_path="iris_model",
        registered_model_name="Iris-dt-model",
        signature=model_signature,
        input_example=X_train.head()
    )

#just to be safe, save a local copy of the model and metrics.
joblib.dump(final_tree, "model.joblib")

with open("metrics.csv", "w") as metrics_file:
    metrics_file.write(f"accuracy,{final_accuracy:.4f}\n")


try:
    client=MlflowClient()
    latest_ver = client.get_latest_versions("Iris-dt-model", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="Iris-dt-model",
        version=latest_ver,
        stage="Production"
    )
    print(f"Model version {latest_ver} moved to Production.")
except Exception as err:
    print("Promotion to Production failed:", err)
