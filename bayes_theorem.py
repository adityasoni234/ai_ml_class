# bayes classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

file_path = "/home/arpit-parekh/Downloads/spam_ham_dataset.csv"

df = pd.read_csv(file_path)
print(df.info())


"""
#   Column      Non-Null Count  Dtype
---  ------      --------------  -----
0   Unnamed: 0  5171 non-null   int64
1   label       5171 non-null   object
2   text        5171 non-null   object
3   label_num   5171 non-null   int64
"""
X = df['text']
y = df["label_num"]

# use CountVectorizer to convert text into tokens/features
vector = CountVectorizer()
X = vector.fit_transform(X)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# predict
text = "How are you?"
text = vector.transform([text])
y_pred = model.predict(text)
print(y_pred)


