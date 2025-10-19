import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/iris.csv")

train, test=train_test_split(data, test_size=0.2, stratify=data['species'], random_state=42)

X_test=test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test =test['species']

model = joblib.load("model.joblib")

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Sanity check / test set accuracy: {acc:.4f}")
sample = X_test.iloc[0:1]
sample_pred = model.predict(sample)
print(f"Sanity check prediction for first sample: {sample_pred[0]}")
