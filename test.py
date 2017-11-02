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
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt') # Download the required nltk data files

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

k_list = list()
max_accuracy = 0

for z in range(3,50):
	if (z % 2 != 0):
		k_list.append(z)


for i in range(1,700):
	bestK = None
	bestKMeanAcc = 0
	ITEMS = i
	
	v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,5),
	                    max_features=ITEMS,
	                    tokenizer=tokenize
	                    )
	#v = CountVectorizer(ngram_range=(1,6),max_features=200, max_df= 0.1)
	
	df_x = v.fit_transform(df.Ingredients.values.astype('U')).todense()
	
	
	for k in k_list:    
	    knn = KNeighborsClassifier(n_neighbors = k, weights='uniform',
	                               algorithm="auto")
	    scores = cross_val_score(knn, df_x, df.target, cv=10)
	
	    print("For k =",k," mean accuracy accross folds =",scores.mean()," standard deviation across folds =",scores.std())
	    if bestK == None or scores.mean() < bestKMeanAcc:
	        bestK = k
	        bestKMeanAcc = scores.mean()
	        
	print("Best k =", k,"with estimated accuracy of",scores.mean())

