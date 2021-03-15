import numpy as np
import pandas as pd
from helpers import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os

current_directory = os.getcwd()
path = current_directory + '/datasets/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
val_res=pd.read_csv(path + 'test.csv')
X_train = train.drop(['label'], axis='columns', inplace=False)
y_train = train['label']

X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.30, random_state=4)

n_components = 9
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
plt.show()
pca.explained_variance_ratio_.sum()

# Already calculated and selected if re-evaluation is needed simply switch the commented out 'param_eval'
# param_eval = {"C": [0.1, 10, 0.5], "gamma": [0.1, 0.01, 0.001], "kernel": ['linear']}
param_eval = {"C": [10], "gamma": [0.1], "kernel": ['rbf']}
rf = SVC()
gs = GridSearchCV(estimator=rf, param_grid=param_eval, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train_pca, y_train)

print(gs.best_score_)
print(gs.best_params_)
best_param = gs.best_params_

t0 = time()
clf = SVC(C=best_param['C'], kernel=best_param['kernel'], gamma=best_param['gamma'])
clf = clf.fit(X_train_pca, y_train)

score = clf.score(pca.transform(X_ts), y_ts)
print('score')
print(score*100)
pred = clf.predict(pca.transform(test))

test['Label'] = pd.Series(pred)
test['ImageId'] = test.index + 1
predicted_results = test[['ImageId', 'Label']]
im_labels=list(test['Label'])
val_res.insert(0,'label',im_labels)
plot_labelsonimages(val_res[0:12])
predicted_results.to_csv('predict.csv', index=False)


