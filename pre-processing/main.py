import pandas as pd
import numpy as np

dataset = pd.read_csv('data.csv')

empty_cols = dataset.isnull().sum()
# print(empty_cols)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(X=x))
# print(x)

le = LabelEncoder()
y = le.fit_transform(y=y)
# print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

import pickle

pickle.dump(model, open('model.pkl', 'wb'))

import matplotlib.pyplot as plt
plt.scatter(x_test[:, 3], y_test, color='red')
plt.plot(x_train[:, 3], model.predict(x_train), color='blue')
plt.title('Plot of data')
plt.xlabel('Age (Standardized)')
plt.ylabel('Purchased (Encoded)')
plt.show()