import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

path = ('/Users/adityasoni234/Desktop/credit_card_customer_data.csv')

df = pd.read_csv(path)

print(df.info())
print(df.head())

X = df.drop(['Sl_No','Customer Key','Total_calls_made'],axis=1)
print(X.head())
print(X.info())


scaler = StandardScaler()
X = scaler.fit_transform(X)


print(X)


model = DBSCAN(eps=0.5)
db = model.fit(X)

labels = db.labels_
print(labels)

unique_labesl = set(labels)
print(unique_labesl)

#plt now
print(X.shape)   # (660, 4)
plt.scatter(X[:,0],X[:,3],c=labels,cmap='rainbow')

plt.title('DBSCAN Clustering')
plt.xlabel('Avg_Credit_Limit')
plt.ylabel('Total_visits_online')
plt.colorbar()
plt.show()
