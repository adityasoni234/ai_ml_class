import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import cv2

File_path = '/Users/adityasoni234/Downloads/data/dataset.csv'

data = pd.read_csv(File_path)

print(data.info())

X = data[["Title","Rating","Reviews","Image URL","Product URL"]]
Y = data["Price"]
input_train,input_test,output_train,output_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = LinearRegression
model.fit = input_train,output_train

model2 = Ridge(alpha=1.0)
model2.fit = input_train,output_train

output_prediction = model.predict(input_test)
output1_prediction = model2.predict(input_test)

print(root_mean_squared_error(output_test,output_prediction))
print(root_mean_squared_error(output_test,output1_prediction))

print("x_value",X)
print("y_value",output_train)

plt.scatter(input_test["Rating"],output_test,color="blue")
plt.scatter(input_test["Rating"],output_test,color="red")
plt.scatter(input_test["Rating"],output_test,color="green")

plt.legend()
plt.xlabel("input")
plt.ylabel("output")
plt.show()