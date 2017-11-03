import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer 


# Load Training Set
dfTrain = pd.read_csv("train.csv", index_col="RecipeId")
trainY = dfTrain.Target.astype('category')

dfTrain = dfTrain.drop(["Target"],axis=1)
df = pd.concat([dfTrain,pd.read_csv("test.csv", index_col="RecipeId")])



v = CountVectorizer(max_features=100)
df_x = v.fit_transform(df.Ingredients.values.astype('U')).todense()

trainX = df_x[:len(dfTrain)]
testX = df_x[len(dfTrain):]

knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm="auto")
knn.fit(trainX,trainY)

predictions = knn.predict(testX)

submissions = pd.read_csv("sampleSubmission.csv", index_col="RecipeId")
submissions.Target = predictions

submissions.to_csv("submission.csv")