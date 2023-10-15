import numpy as np
import pandas as pd
import joblib
from src.features.build_features import edit_dataframe

model = joblib.load("../../models/model.pkl")

df = pd.read_csv("../../data/raw/test.csv")
df = edit_dataframe(df)
df.head()

y_pred = model.predict(df)

sdf = pd.DataFrame()
sdf["ID"] = df.ID
sdf["target"] = y_pred
sdf["target"][sdf["target"] < 0] = 0
sdf['target'] = sdf['target'].map(lambda x: int(x))

sdf.to_csv("../../data/processed/submission.csv")