import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

xtrain = np.load(open('hollywood_Xtr.npy'))
ytrain = np.load(open('hollywood_Ytr.npy'))

mms = MinMaxScaler().fit(xtrain)
xtrain = mms.transform(xtrain)

xtest = np.load(open('hollywood_Xte.npy'))
ytest = np.load(open('hollywood_Yte.npy'))

xtest = mms.transform(xtest)

print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

svm = SVC(kernel='linear').fit(xtrain, ytrain)
preds = svm.predict(xtest)
print confusion_matrix(ytest, preds)
print classification_report(ytest, preds)
print svm.score(xtest, ytest)
