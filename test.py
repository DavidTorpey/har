import numpy as np
from glob import glob
from scipy.io import loadmat as lm
from sklearn.svm import LinearSVC as svm
import sys

def norm(v):
	v = v / np.linalg.norm(v, ord=1)
	return np.sign(v) * np.sqrt(np.abs(v))

actions = ['Diving',
           'Golf-Swing',
           'Kicking',
           'Lifting',
           'Riding-Horse',
           'Run',
           'SkateBoarding',
           'Swing-Bench',
           'Swing-Side',
           'Walk']


def get_action(f):
    for e in actions:
        if e.lower() in f.lower():
            return e

ii = int(sys.argv[1])

accs = []
for i in range(1, ii+1):
	fs = glob('loocv/fold_' + str(i) + '/train*')
	X_tr = np.array([norm(lm(e)['encoding'].ravel()) for e in fs])
	Y_tr = np.array([get_action(e) for e in fs])
	fs = glob('loocv/fold_' + str(i) + '/test*')
	X_te = np.array([norm(lm(e)['encoding'].ravel()) for e in fs])
	Y_te = np.array([get_action(e) for e in fs])
	
	
	m = svm().fit(X_tr, Y_tr)
	acc = m.score(X_te, Y_te)

	print i, acc

	accs.append(acc)
accs = np.array(accs)

print 'Acc', accs.mean()
