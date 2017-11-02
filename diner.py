#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:54:09 2017

@author: cwalsh
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv( "train.csv", index_col="RecipeId")
df.target = df.Target.astype('category')
df.head()

import operator

ITEMS = 20
cuisines=["","French","Italian","Indian","Chinese","Thai","Greek","Mexican"]
#

v = CountVectorizer(ngram_range=(1,6),max_features=ITEMS, max_df= 0.1)


for i in range(1,8):
    df_x = v.fit_transform(df[df.Target == i].Ingredients.values.astype('U'))
    print("\nTop",ITEMS,"Ingredients for",cuisines[i],"Recipes")
    n = 1
    for ingr in sorted(v.vocabulary_.items(), key=operator.itemgetter(1)):
        print(n,ingr[0])
        n += 1
        
        
ITEMS = 200
v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3),max_features=ITEMS)

df_x = v.fit_transform(df.Ingredients.values.astype('U'))
idf = v.idf_

topIngredients = pd.DataFrame({"Ingredient": v.get_feature_names(),"TfIdf":idf})
print("Top",ITEMS,"Ingredients")
topIngredients.sort_values(by="TfIdf",ascending=False)


import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt') # Download the required nltk data files

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems



df_x = v.fit_transform(df.Ingredients.values.astype('U'))
idf = v.idf_

topIngredients = pd.DataFrame({"Ingredient": v.get_feature_names(),"TfIdf":idf})

print("Top",ITEMS,"Ingredients")
topIngredients.sort_values(by="TfIdf",ascending=False)

bestK = None
bestKMeanAcc = 0
ITEMS = 500

v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,5),
                    max_features=ITEMS,
                    #max_df=0.8,
                    tokenizer=tokenize

                    )
#v = CountVectorizer(ngram_range=(1,6),max_features=200, max_df= 0.1)

df_x = v.fit_transform(df.Ingredients.values.astype('U')).todense()

for k in [13]:    
    knn = KNeighborsClassifier(n_neighbors = k, weights='uniform',
                               algorithm="auto")
    scores = cross_val_score(knn, df_x, df.target, cv=10)

    print("For k =",k," mean accuracy accross folds =",scores.mean()," standard deviation across folds =",scores.std())
    if bestK == None or scores.mean() < bestKMeanAcc:
        bestK = k
        bestKMeanAcc = scores.mean()
        
print("Best k =", k,"with estimated accuracy of",scores.mean())