import os 
import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

def img_info(folder):
    images = []
    labels = []

    for subfolder in os.listdir(folder):
        
        if os.path.isdir (os.path.join(folder,subfolder)):
        
            for img in os.listdir(os.path.join(folder,subfolder)):
                
                if(img.endswith(".jpg") or img.endswith(".jpeg")):

                    modelimage = cv2.imread(os.path.join(folder,subfolder,img))

                if modelimage is not None: 
                    try:
                        modelimage = cv2.resize(modelimage,(100,100))
                        modelimage = modelimage.flatten()
                        images.append(modelimage)
                        labels.append(subfolder)
                    except Exception as e:

                        print(f"Error in processing {img},{str(e)}")

                else:
                    print(f"could not load image :{img}")

    return images,labels

images,labels = img_info("/Users/adityasoni234/Desktop/animals")
print(images)
print(labels)

X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size=0.2,random_state=42,stratify=labels)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

def plot_images(images,true_labels,predicted_labels,min_size=9):
    plt.figure(figsize=(10,10))
    for i in range(min(len(images),min_size)):
        plt.subplot(3,3,i+1)

        images[i] = cv2.cvtColor(images[i].reshape(100, 100, 3), cv2.COLOR_BGR2RGB)
        plt.imshow(images[i])  # unflaatten the image

        plt.title(f"True: {true_labels[i]} \n Predicted:{predicted_labels[i]}")
        plt.axis("off")
    plt.show()

plot_images(X_test,y_test,y_pred)


modelImage = cv2.imread("/Users/adityasoni234/Desktop/animals/butterfly/1c1de9b3a2.jpg")

if modelImage is not None:
     modelImage = cv2.resize(modelImage,(100,100))
     modelImage = modelImage.flatten()
     modelImage = modelImage.reshape(1, -1)  # Reshape for single prediction
     y_pred = knn.predict(modelImage)
     print(y_pred)
else:
     print("Error: Could not load the image. Please check if the file path is correct.")



