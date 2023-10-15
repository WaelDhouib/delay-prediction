# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime as dt

# Reading CSV file
df = pd.read_csv("../../data/raw/train.csv")
df.head()

# Formatting dates
df['STD'] = df['STD'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))