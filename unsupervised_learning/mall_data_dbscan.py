import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


path = '/Users/adityasoni234/Desktop/Mall_Customers.csv'

df = pd.read_csv(path)
print(df.info())

df['Genre'] = df['Genre'].replace({'Male': 0, 'Female': 1})

X = df.drop(['CustomerID'],axis=1)
print(X.info())

scale = StandardScaler()
X = scale.fit_transform(X)

print(X)


model = DBSCAN()
db = model.fit(X)

labels = db.labels_
print(labels)

unique_labels = set(labels.tolist())

legend_labels = ['Cluster ' + str(label) for label in unique_labels]
print(legend_labels)



dots = plt.scatter(X[:,2],X[:,3],c=labels,cmap='rainbow')

print(dots.legend_elements()[0])

plt.scatter(X[:,2],X[:,3],c=labels,cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar()
plt.legend(dots.legend_elements()[0],legend_labels,loc='upper right')
plt.show()



"""
The dots.legend_elements()[0] returns the colored markers/handles that will appear in the legend, while legend_labels provides the text descriptions. Both are needed to create a complete legend that shows:

The colored dot/marker (handle)
The corresponding cluster label text (legend_labels)
"""
