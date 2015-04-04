import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import svm
import pandas as pd
import numpy as np
import pickle


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

print "Trying to construct a SVM classifier"
model=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,
shrinking=True, tol=0.001, verbose=False)

print "Wrote the model for the SVM classifier"
# print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc'))

print "Retrain on all training data, predicting test labels...\n"
model.fit(X,y)
print "Model Fitted"
result = model.predict_proba(X_test)[:,1]
print "Model Predicted"
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

print "Model outputted"
# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_svm.csv'), index=False, quoting=3)
print "Wrote results to Bag_of_Words_model.csv"	