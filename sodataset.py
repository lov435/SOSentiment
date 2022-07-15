# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 23:26:49 2022

@author: viral
"""
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Dataset:
    def loadData(self):
        df = pd.read_csv('survey1.csv', encoding="ISO-8859-1")
        #print(df['Comment'].to_string()) 
        
        stemmer = PorterStemmer()
        words = stopwords.words("english")
        df['clean_comment'] = df['Comment'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
        print(df['clean_comment'].to_string()) 

        vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
        final_features = vectorizer.fit_transform(df['clean_comment']).toarray()
        print(final_features.shape)
        
        X = df['clean_comment']
        Y = df['Weakness Category']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
        
        # instead of doing these steps one at a time, we can use a pipeline to complete then all at once
        pipeline = Pipeline([('vect', vectorizer),
                             ('chi',  SelectKBest(chi2, k='all')),
                             ('clf', RandomForestClassifier())])
        
        # fitting our model and save it in a pickle for later use
        model = pipeline.fit(X_train, y_train)
        with open('RandomForest.pickle', 'wb') as f:
            pickle.dump(model, f)
        
        ytest = np.array(y_test)
        
        # confusion matrix and classification report(precision, recall, F1-score)
        print(classification_report(ytest, model.predict(X_test)))
        print(confusion_matrix(ytest, model.predict(X_test)))

if __name__ == '__main__':
    d = Dataset()
    d.loadData()