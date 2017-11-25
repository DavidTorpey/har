from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

X = np.load(open('X_ucf.npy'))
Y = np.load(open('Y_ucf.npy'))

loo = LeaveOneOut(len(X))
accuracies = []
ypred = []
ytrue = []
for train_idx, test_idx in loo:
	xtrain = X[train_idx, :]
	ytrain = Y[train_idx]
	
	mms = MinMaxScaler().fit(xtrain)
	xtrain = mms.transform(xtrain)

	xtest = X[test_idx, :]
	ytest = Y[test_idx]
	
	xtest = mms.transform(xtest)

	svm = SVC(kernel='linear').fit(xtrain, ytrain)
	score = svm.score(xtest, ytest)
	ytrue.append(ytest[0])
	ypred.append(svm.predict(xtest)[0])

	print score
	accuracies.append(score)

print np.array(accuracies).mean()
print confusion_matrix(ytrue, ypred)
print classification_report(ytrue, ypred)
