import numpy as np

a = [1,2,3,4,5]
b = [6,7,8,9,10]

random_one = np.random.choice(a)
random_two = np.random.choice(b)
print(random_one, random_two)

# a = np.array([10, 20, 30, 40])
# b = np.array([10, 20, 30, 40])

# # print(np.sum(a+b))
# # print(a[1])
# # arr = np.arange(1,12)
# # print(arr)
# # print(a*2)

# arr_2d = np.arange(1,9).reshape((2,2,2))
# # print(arr_2d)
# arr_3d = np.arange(1,10).reshape(3,3)
# print(arr_3d)
# print(arr_3d.T)
# print(np.var(a))

# import matplotlib.pyplot as plt

# x = [1,2,3,4,5]
# y = [10,20,30,40,50]
# plt.scatter(x,y, color='blue')
# plt.xlabel("study_hours")
# plt.ylabel("score")
# plt.title("Random")
# plt.grid("true")
# # plt.show()

# from sklearn.linear_model import LinearRegression

# x = [[1], [2], [3], [4], [5]]
# y = [10, 20, 30, 40, 50]
# model = LinearRegression()
# model.fit(x, y)
# # print(model.predict([[6]]))

# import pandas as pd

# data = {
#     "name": ["Alice", "Bob", "Charlie"],
#     "age": [25, 30, 35],
#     "score": [85, 90, 95]
# }

# df = pd.DataFrame(data)
# print(df)
# # df.to_csv("output.csv", index=False)
# print(df.describe())