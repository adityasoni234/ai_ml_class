import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes
data=load_diabetes()
print(data)

x=data.data
y=data.target

feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
df=pd.DataFrame(data.data,columns=feature_names)
df['Target']=data.target
print(df)
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=24)
model=ElasticNet(alpha=1.0,l1_ratio=0.5)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print(y_predict)
print(root_mean_squared_error(y_test,y_predict))
print(x_test.shape)
print(y_test.shape)

plt.scatter(x_test[:,0],y_test,color="blue")
plt.scatter(x_test[:,0],y_predict,color='red')
#plt.scatter(y_test,y_predict,color='green')
plt.show()