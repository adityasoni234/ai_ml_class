import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster  import KMeans
from sklearn.datasets import make_blobs


x,y  = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_pred = kmeans.predict(x)

print(y)

accuracy = accuracy_score(y, y_pred)
print(accuracy)


plt.scatter(x[:,0],x[:,1],c=y_pred,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='p',color='black')
plt.title('K_Means')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()