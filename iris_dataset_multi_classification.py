import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score

#load the iris dataset
iris = load_iris()
print(iris.DESCR)

x = iris.data
y = iris.target

print("shape of X:",x.shape)
print("shape of Y:",y.shape)

#split the dataset into training and testing sets
x_train ,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

colors = ["red","blue","black"]


for i in range(3):
    plt.scatter(x[y==i,0],x[y==i,1],label=iris.target_names[i],color=colors[i])


plt.title('iris dataset')
plt.show()