import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = '/Users/adityasoni234/Downloads/test.csv'
data = pd.read_csv(file_path)
#data.head()

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
print(data.info())

X = data[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
Y = data["Name"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42) 

model = LogisticRegression()
model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test,Y_predict)
print("accuracy :",accuracy)

