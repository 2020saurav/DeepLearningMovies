import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import svm
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression	
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]  

with open("a-file.pickle","r") as f:
	X_all=pickle.load(f)

lentrain = 25000
X = X_all[:lentrain]
X_test = X_all[lentrain:]

# print X[1]
# print "-----------"
# print X[2]
# print "-----------"

# model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
#                          C=1, fit_intercept=True, intercept_scaling=1.0, 
#                          class_weight=None, random_state=None)


logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True ,n_components=100,n_iter=1)

model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])



print "Trying to construct a Pipeline classifier"

print "Wrote the model for the Pipeline classifier"
# print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc'))

print "Retrain on all training data, predicting test labels...\n"
# X_dense=X.todense()

model.fit(X,y)
print "Model Fitted"
# X_test_dense=X_test.todense()
result = model.predict_proba(X_test)[:,1]
print "Model Predicted"
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

print "Model outputted"
# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_pipeline.csv'), index=False, quoting=3)
print "Wrote results to Bag_of_Words_model.csv"	