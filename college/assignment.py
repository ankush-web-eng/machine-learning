data = {
    "Patient_ID": [101, 102, 103, 104, 105, 106, 107],
    "Age": [25, 30, 28, None, 45, 50, 29],
    "Weight": [58, 75, None, 55, 85, 90, None],
    "BloodPressure": [120, None, 130, 110, 140, 150, None],
    "Cholesterol": [200, 210, 220, 190, None, 250, 230],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Male", "Female"]
}

import pandas as pd

df = pd.DataFrame(data)
print(df.isnull().sum())
drop_df = df.dropna()
df.to_csv("patients.csv", index=False)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Weight'].fillna(df['Weight'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].mean(), inplace=True)

print(df)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

mm = MinMaxScaler()
ss = StandardScaler()

df[['mm_Age', 'mm_Weight', 'mm_BloodPressure', 'mm_Cholesterol']] = mm.fit_transform(df[['Age', 'Weight', 'BloodPressure', 'Cholesterol']])
df[['ss_Age', 'ss_Weight', 'ss_BloodPressure', 'ss_Cholesterol']] = ss.fit_transform(df[['Age', 'Weight', 'BloodPressure', 'Cholesterol']])

print(df)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df = df.join(pd.DataFrame(ohe.fit_transform(df[['Gender']]).toarray(), columns=ohe.get_feature_names_out(['Gender'])))

print(df)