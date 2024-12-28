import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

df['target'] = iris.target
print(df)

x=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)



accuracy = accuracy_score(y_test,y_pred)
print("accuracy :",accuracy)
