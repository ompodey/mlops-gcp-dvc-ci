import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier as DC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
#ok

data = pd.read_csv("data/iris.csv")



train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species


model = DC(max_depth = 3, random_state = 1)
model.fit(X_train,y_train)
pred= model.predict(X_test)
acc=accuracy_score( pred, y_test)
print(f"Model accuracy: {acc:.4f}")

joblib.dump(model, "model.joblib")

with open("metrics.csv", "w") as f:
    f.write("accuracy,%.4f\n" % acc)
