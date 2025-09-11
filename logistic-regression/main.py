import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
# print(df)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print("\nx_train: \n", x_train)
# print("\nx_test: \n", x_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
print(classifier.predict(sc.transform([[30, 87000]])))