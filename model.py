import numpy as py
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
loaded_model = pickle.load(open('F:/sem6/ML/lab7/trained_model.sav', 'rb'))


count = CountVectorizer()
spam = pd.read_csv('F:/sem6/ML/sms_spam/spam.csv')
spam = spam[['v1', 'v2']]
spam = spam.rename(columns={'v1': 'label', 'v2': 'text'})
X_train, X_test, y_train, y_test = train_test_split(spam['text'], spam['label'], test_size=0.2, random_state=42)
test=['Quality assurance is required to make sure that the software system works according to the requirements. Were all the features implemented as agreed? Does the program behave as expected? All the parameters that you test the program against should be stated in the technical specification document.']
X_train_count= count.fit_transform(X_train)

lr_count=LogisticRegression()
lr_count.fit(X_train_count,y_train)
X_train_count=count.transform(X_test)

X_test_new=count.transform(test)
Y_test_new=loaded_model.predict(X_test_new)
# score=accuracy_score(y_test, Y_test_new)
print("result of X_test ",Y_test_new)