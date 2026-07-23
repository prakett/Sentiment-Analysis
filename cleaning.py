import pandas as pd

train = pd.read_csv("dataset2/train.csv")
validation = pd.read_csv("dataset2/validation.csv")
test = pd.read_csv("dataset2/test.csv")

train.fillna("", inplace=True)
validation.fillna("", inplace=True)
test.fillna("", inplace=True)

train.to_csv("dataset3/train.csv", index=False)
validation.to_csv("dataset3/validation.csv", index=False)
test.to_csv("dataset3/test.csv", index=False)

print("Missing values replaced and datasets saved successfully!")