from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
import numpy as np

X = np.load(open('X_ucf.npy'))
Y = np.load(open('Y_ucf.npy'))

loo = LeaveOneOut(len(X))
accuracies = []
for train_idx, test_idx in loo:
	xtrain = X[train_idx, :]
	ytrain = Y[train_idx]
	xtest = X[test_idx, :]
	ytest = Y[test_idx]
	
	svm = SVC(kernel='linear').fit(xtrain, ytrain)
	score = svm.score(xtest, ytest)
	print score
	accuracies.append(score)

print np.array(accuracies).mean()	
