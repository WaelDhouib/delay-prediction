import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

df = pd.read_csv("../../data/interim/train.csv")

X = df.drop(['ID','target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=565654)

model = xgb.XGBRegressor(n_estimators=10000,
                       nthread=4,
                       min_child_weight=6,
                       learning_rate=0.5,
                       subsample=0.7,
                       max_depth=6,
                       colsample_bytree=0.6,
                       tree_method='exact')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

joblib.dump(model, "../../models/model.pkl")
