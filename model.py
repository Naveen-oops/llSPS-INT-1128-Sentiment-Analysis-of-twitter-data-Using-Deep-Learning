# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:50:15 2020

@author: NAGAMANIKANTA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv', encoding = "ISO-8859-1",nrows=20000)
import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data=[]
for i in range(0,19999):
        print(i)
        review=df['SentimentText'][i]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=30000)
X = cv.fit_transform(data).toarray()
y=df.iloc[:-1,1].values
import pickle
pickle.dump(cv, open("cv.pkl", "wb"))
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = 22828))

model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))

model.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

X_train=X
y_train=y

model.fit(X_train,y_train,batch_size=128,epochs=30)

model.save('model.h5')

    