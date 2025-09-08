import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('data.csv')

empty_cols = dataset.isnull().sum()
print(empty_cols)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)
print(y)