import numpy as np
import pandas as pd

df = pd.read_csv("../../data/raw/train.csv")

df['STD'] = df['STD'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

df["DAY"] = ""
for i in range(len(df["STD"])):
    df["DAY"][i] = a[i].weekday()


X = df.drop(['ID','target'], axis=1)
y = df['target']

X['DATOP'] = X['DATOP'].map(lambda x: x[5:])
X['DATOP'] = X['DATOP'].map(lambda x: x.replace('-',''))
X['DATOP'] = pd.to_numeric(X['DATOP'])

X['STD'] = X['STD'].map(lambda x: x[-9:])
X['STD'] = X['STD'].map(lambda x: x[:-3])
X['STD'] = X['STD'].map(lambda x: x.replace(':',''))
X['STD'] = pd.to_numeric(X['STD'])

X['STA'] = X['STA'].map(lambda x: x[-9:])
X['STA'] = X['STA'].map(lambda x: x[:-3])
X['STA'] = X['STA'].map(lambda x: x.replace(':',''))
X['STA'] = pd.to_numeric(X['STA'])

X["STATUS"] = X["STATUS"].replace(dict(zip(X["STATUS"].unique(),range(0,len(X["STATUS"].unique())))));
X["DEPSTN"] = X["DEPSTN"].replace(dict(zip(X["DEPSTN"].unique(),range(0,len(X["DEPSTN"].unique())))));
X["ARRSTN"] = X["ARRSTN"].replace(dict(zip(X["ARRSTN"].unique(),range(0,len(X["ARRSTN"].unique())))));
X["AC"] = X["AC"].replace(dict(zip(X["AC"].unique(),range(0,len(X["AC"].unique())))));
X["FLTID"] = X["FLTID"].replace(dict(zip(X["FLTID"].unique(),range(0,len(X["FLTID"].unique())))));

X['DAY'] = pd.to_numeric(X['DAY'])

X['DAY'] = pd.to_numeric(X['DAY'])

X.to_csv("../../data/interim/train.csv")