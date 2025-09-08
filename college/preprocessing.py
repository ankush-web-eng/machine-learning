import pandas as pd
import numpy as np

data = {
    "age" : [27,28,29,None, 30,31,None, 32,33,40],
    "height" : [160, 161, 162, 163, None, 165, 166, 167, 168, 169],
    "weight" : [55, 60, 65, None, 70, 75, 80, 85, 90, 95],
    "city" : ["New York", "Los Angeles", "Chicago", "Houston", None,
                "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas"],
    "marks" : [80, 90, 88, None, 85, 95, 78, 88, 90, 92]
}

df = pd.DataFrame(data)
# print(df.isnull().sum())
# df_drop = df.dropna()

df['age'].fillna(df['age'].mean(), inplace=True)
df['height'].fillna(df['height'].mean(), inplace=True)
df['weight'].fillna(method='ffill', inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)
df['marks'].fillna(method='bfill', inplace=True)

print(df)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mm = MinMaxScaler()
ss = StandardScaler()

df[['mm_age', 'mm_height', 'mm_weight']] = mm.fit_transform(df[['age', 'height', 'weight']])
df[['ss_age', 'ss_height', 'ss_weight']] = ss.fit_transform(df[['age', 'height', 'weight']])

print(df)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[['city']])
ohe_columns = ohe.get_feature_names_out(['city'])
df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns)
print(df_ohe)

le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
print(df)

# x = df[["age", "height", "weight", "city"]]
# y = df["marks"]

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(x_train, y_train)

# predictions = model.predict(x_test)
# print("Predictions:", predictions)
# print("Prediction Score:", model.score(x_test, y_test))