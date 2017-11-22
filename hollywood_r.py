import numpy as np
from sklearn.svm import SVC

xtrain = np.load(open('hollywood_Xtr.npy'))
ytrain = np.load(open('hollywood_Ytr.npy'))
xtest = np.load(open('hollywood_Xte.npy'))
ytest = np.load(open('hollywood_Yte.npy'))

print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

svm = SVC(kernel='linear').fit(xtrain, ytrain)
print svm.score(xtest, ytest)
