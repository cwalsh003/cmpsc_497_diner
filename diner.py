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

df = pd.read_csv( "train.csv", index_col="RecipeId")
df.target = df.Target.astype('category')
df.head()

import operator

ITEMS = 15
cuisines=["","French","Italian","Indian","Chinese","Thai","Greek","Mexican"]
v = CountVectorizer(ngram_range=(1,3),max_features=ITEMS)
for i in range(1,8):
    df_x = v.fit_transform(df[df.Target == i].Ingredients.values.astype('U'))
    print("\nTop",ITEMS,"Ingredients for",cuisines[i],"Recipes")
    n = 1
    for ingr in sorted(v.vocabulary_.items(), key=operator.itemgetter(1)):
        print(n,ingr[0])
        n += 1
        
        
ITEMS = 50
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

ITEMS = 50
v = TfidfVectorizer(sublinear_tf=True, tokenizer=tokenize, stop_words='english', ngram_range=(1,3),max_features=ITEMS)
df_x = v.fit_transform(df.Ingredients.values.astype('U'))
idf = v.idf_

topIngredients = pd.DataFrame({"Ingredient": v.get_feature_names(),"TfIdf":idf})
print("Top",ITEMS,"Ingredients")
topIngredients.sort_values(by="TfIdf",ascending=False)