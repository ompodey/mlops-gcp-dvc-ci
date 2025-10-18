import pandas as pd

data = pd.read_csv("data/iris.csv")

rows_to_delete=data.sample(n=50, random_state=42)

data= data.drop(rows_to_delete.index)

data.to_csv("data/iris.csv", index=False)

print("rows in new data : ",len(data)")