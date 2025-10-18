import pandas as pd

data = pd.read_csv("data/iris.csv")
datav2 = data.sample(n=50, random_state=42)
data = pd.concat([data, datav2], ignore_index=True)
data.to_csv("data/iris.csv", index=False)

print(f"New dataset rows: {len(data)}")