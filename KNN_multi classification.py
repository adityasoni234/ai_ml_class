import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine,load_diabetes,load_iris
import pandas as pd

# Load the dataset
# wine = load_wine()
# print(wine.DESCR)
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data,columns=diabetes.feature_name)
print(df.head())
df['target'] = diabetes.target
print(df['target'])

# label the target data on 3 categories [0,1,2]
df['newTarget'] = np.where(df['target']<100,0,np.where(df['target']<200,1,2))

# split the data into training and testing sets
# take all the other  values as features

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target','newTarget'],axis=1),df['newTarget'],test_size=0.2,random_state=42)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = knn.predict(X_test)
y_pred2 = logistic.predict(X_test)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy)
print("Accuracy:", accuracy2)

# plot the data
plt.title('KNN - Multiclass Classification')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.scatter(X_test['age'],X_test['bp'],c=y_pred,cmap='rainbow')
plt.colorbar()
plt.show()