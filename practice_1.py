import pandas as pd
import numpy as np 
import matplotlib as plt
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

File_path = '/Users/adityasoni234/Downloads/weather_forecast_data.csv'

data = pd.read_csv(File_path)

print(data.info())

X = data[["Temperature","Humidity","Wind_Speed","Cloud_Cover","Pressure"]]
Y = data["Rain"]
input_train,input_test,output_train, output_test = train_test_split(X,Y, test_size=0.2,random_state=42)

label_encoder = LabelEncoder()
output_train = label_encoder.fit_transform(output_train)

model = LinearRegression()
model.fit(input_train,output_train)

model1 =Ridge(alpha=1.0)
model1.fit(input_test,output_train)

output_prediction = model.predict(input_test)
output_prediction1 = model1.predict(input_test)

print(root_mean_squared_error(output_test,output_prediction))
print(root_mean_squared_error(output_test,output_prediction1))

print("x_value",X)
print("y_value",output_train)
