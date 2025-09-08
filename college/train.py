import pandas as pd

df = pd.read_csv("50_Startups.csv")
# print(df.describe())

x = df[["R&D Spend", "Administration", "Marketing Spend"]]
y = df[["Profit"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("Predictions:", predictions)
print("Prediction Score:", model.score(x_test, y_test))

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

from matplotlib import pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Actual vs Predicted Profits")
plt.show()

# data = {
#     "experience" : [5,6,7,7,8,9,10,11,12,13,14],
#     "salary" : [50000,60000,70000,70000,80000,90000,100000,110000,120000,130000,140000]
# }

# df  = pd.DataFrame(data)
# # print(df.head())
# # print(df.describe())
# # df.to_csv("salary_data.csv", index=False)

# # df = pd.read_csv("salary_data.csv")
# # print(df.head())

# x = df[["experience"]]
# y = df[["salary"]]
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(x_train, y_train)

# predictions = model.predict(x_test)
# print(predictions)