import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


path = '/Users/adityasoni234/Desktop/Mall_Customers.csv'

df = pd.read_csv(path)
print(df.info())

X = df.drop(['CustomerID'],axis=1)
print(X.info())

scale = StandardScaler()
X = scale.fit_transform(X)

print(X)


model = DBSCAN()
db = model.fit(X)

labels = db.labels_
print(labels)

