import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import root_mean_squared_error

file_path = '/Users/adityasoni234/Downloads/gym.csv'
data = pd.read_csv(file_path)

print(data.info())
X = data[["Age","Weight (kg)","Height (m)","Max_BPM","Avg_BPM","Resting_BPM","Session_Duration (hours)","Calories_Burned","Fat_Percentage","Workout_Frequency (days/week)","Experience_Level","BMI"]]
output = data["Water_Intake (liters)"]
input_train,input_test,output_train,output_test=train_test_split(X,output,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(input_train,output_train)

model2 = Ridge(alpha=1.0)
model2.fit(input_train,output_train)

output_prediction = model.predict(input_test)
output1_prediction = model2.predict(input_test)

print(root_mean_squared_error(output_test,output_prediction))
print(root_mean_squared_error(output_test,output1_prediction))

print("x value",X)
print("y value",output_train)

plt.scatter(input_test["Calories_Burned"], output_test, color="blue", label="Original Data")
plt.scatter(input_test["Calories_Burned"],output_prediction, color="red", label="Linear Regression")
plt.scatter(input_test["Calories_Burned"],output1_prediction, color="green", label="Ridge Regression")
plt.legend()
plt.xlabel("input")
plt.ylabel("output")
plt.show()