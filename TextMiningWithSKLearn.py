# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:10:57 2019

@author: Sai Manideep
"""

import pandas as pd

''' Importing data'''

import os
def data2df (path, label):
    file, text = [], []
    for f in os.listdir(path):    
        file.append(f)
        fhr = open(path+f, 'r', encoding='utf-8', errors='ignore') 
        t = fhr.read()
        text.append(t)
        fhr.close()
    return(pd.DataFrame({'file': file, 'text': text, 'class':label}))

dfNonPro = data2df('HealthProNonPro/NonPro/', 0)
dfPro = data2df('HealthProNonPro/Pro/', 1)

df = pd.concat([dfPro, dfNonPro], axis=0)

#print (df.sample(frac=0.005))

''' setting up the data '''
X, y = df['text'], df['class']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)

''' Custom preprocessor
    Doing what has been suggested in the research paper (except for the porter's stemming algorithm) 
'''
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def preprocess(text):
    # replace one or more white-space characters with a space
    regex = re.compile(r"\s+")                               
    text = regex.sub(' ', text)    
    # lower case
    text = text.lower()          
    # remove digits and punctuation
    regex = re.compile(r"[%s%s]" % (string.punctuation, string.digits))
    text = regex.sub(' ', text)           
    # remove stop words
    sw = stopwords.words('english')
    text = text.split()                                              
    text = ' '.join([w for w in text if w not in sw]) 
    # remove short words
    ' '.join([w for w in text.split() if (len(w) >= 3 and len(w) <= 60)])
    # lemmatize
    text = ' '.join([(WordNetLemmatizer()).lemmatize(w) for w in text.split()]) 
    return text

''' Setting up model pipeline '''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf = Pipeline(steps=[
    ('pp', TfidfVectorizer(
        preprocessor=preprocess,
        use_idf=True, smooth_idf=True,
        min_df=1, max_df=1.0, max_features=None, 
        ngram_range=(1, 1))),
    ('mdl',     MultinomialNB())
    ])
    
''' Setting up grid search '''
from sklearn.model_selection import GridSearchCV
param_grid = {
    'mdl__alpha':[0.01, 0.1, 0.2, 0.5, 1],
    'pp__norm': ['l1', 'l2', None]
}
gscv = GridSearchCV(clf, param_grid, iid=False, cv=4, return_train_score=False)

gscv.fit(Xtrain, ytrain)
#print(gscv.best_params_, "\n")

''' Evaluation of model '''
ypred = gscv.best_estimator_.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))