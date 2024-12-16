import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

#classes are more then 2
# we will use logisticregression for multi classification 

# [weight,height]
# [cat ,dog ,rabbit]

#in classification input should always be in 2D array

data = np.array([
        [5,20],#cat
        [3,12],#cat
        [4,25],#cat
        [6,21],#rabbit
        [7,30],#rabbit
        [10,40],#dog
        [12,50],#dog
        [20,65],#dog
])

target = np.array([0,0,0,1,1,2,2,2])

model = LogisticRegression()
model.fit(data,target)
#[1] cat
#[2] rabbit
#[3] dog
print(model.predict([[12,50]]))

colors = ["red","blue","green"]
labels = ["cat","rabbit","dog"]

for i in range(3):
    plt.scatter(data[target==i,0],data[target==i,1],label=labels[i],color=colors[i])

plt.xlabel('weight')
plt.ylabel('height')
plt.title('classification using logistic regration ')
plt.legend()
plt.grid()
plt.show()