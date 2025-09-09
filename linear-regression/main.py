import pandas as pd
import numpy as np

dataset = pd.read_csv('SalaryData.csv')
# print(dataset)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# print("\nx_train: \n", x_train)
# print("\nx_test: \n", x_test)
# print("\ny_train: \n", y_train)
# print("\ny_test: \n", y_test)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train[:, :] = sc.fit_transform(x_train[:, :])
# x_test[:, :] = sc.transform(x_test[:, :])

print("\n x_train: \n", x_train)
print("\n x_test: \n", x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, colorizer='red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Salary vs Years of Experience Plot (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of Employees')
plt.show()