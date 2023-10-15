import numpy as np
import pandas as pd
import datetime as dt

def edit_dataframe():
    df['STD'] = df['STD'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    df["DAY"] = ""
    for i in range(len(df["STD"])):
        df["DAY"][i] = df['STD'][i].weekday()
        
    df.dtypes

    df['DATOP'] = df['DATOP'].map(lambda x: x[5:])
    df['DATOP'] = df['DATOP'].map(lambda x: x.replace('-',''))
    df['DATOP'] = pd.to_numeric(df['DATOP'])

    df['STD'] = df['STD'].astype(str)
    df['STD'] = df['STD'].map(lambda x: x[-9:])
    df['STD'] = df['STD'].map(lambda x: x[:-3])
    df['STD'] = df['STD'].map(lambda x: x.replace(':',''))
    df['STD'] = pd.to_numeric(df['STD'])

    df['STA'] = df['STA'].map(lambda x: x[-9:])
    df['STA'] = df['STA'].map(lambda x: x[:-3])
    df['STA'] = df['STA'].map(lambda x: x.replace(':',''))
    df['STA'] = pd.to_numeric(df['STA'])

    df["STATUS"] = df["STATUS"].replace(dict(zip(df["STATUS"].unique(),range(0,len(df["STATUS"].unique())))));
    df["DEPSTN"] = df["DEPSTN"].replace(dict(zip(df["DEPSTN"].unique(),range(0,len(df["DEPSTN"].unique())))));
    df["ARRSTN"] = df["ARRSTN"].replace(dict(zip(df["ARRSTN"].unique(),range(0,len(df["ARRSTN"].unique())))));
    df["AC"] = df["AC"].replace(dict(zip(df["AC"].unique(),range(0,len(df["AC"].unique())))));
    df["FLTID"] = df["FLTID"].replace(dict(zip(df["FLTID"].unique(),range(0,len(df["FLTID"].unique())))));

    df['DAY'] = pd.to_numeric(df['DAY'])

    return df

df = pd.read_csv("../../data/raw/train.csv")
df = edit_dataframe(df)
df.head()
df.to_csv("../../data/interim/train.csv")