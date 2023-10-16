import numpy as np
import pandas as pd
import joblib
import datetime as dt

model = joblib.load("../../models/model.pkl")

df = pd.read_csv("../../data/raw/test.csv", index_col=False)

df.head()
df.dtypes

sdf = pd.DataFrame()
sdf["ID"] = df.ID

df = df.drop(['ID'], axis=1)
df.index.name = None
y_pred = model.predict(df)

sdf["target"] = y_pred
sdf["target"][sdf["target"] < 0] = 0
sdf['target'] = sdf['target'].map(lambda x: int(x))
sdf.describe()

sdf.to_csv("../../data/processed/submission.csv", index=False)