# -*- coding: utf-8 -*-
"""
Titanic competition on Kaggle

"""
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('D:/DDocuments/Python/Kaggle/Titanic_comp/input/train.csv')
test_data = pd.read_csv('D:/DDocuments/Python/Kaggle/Titanic_comp/input/test.csv')

# looking at the data
train_data.head()# for testing
train_data.info()
train_data.describe()
train_data["Cabin"].value_counts()# looks at data types



import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20,15))
plt.show()

# model
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass",  "Age", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# drop data with missing columns
cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]

reduced_X = X.drop(cols_with_missing,axis=0)
reduced_y = y.drop(cols_with_missing, axis=0)

model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1)

model.fit(reduced_X,reduced_y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('My_submission3.csv', index=False)
print("Your submission was successfully saved!")
