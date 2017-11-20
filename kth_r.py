import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from glob import glob

def get_person(f):
        temp = f.split('/')[-1]
        return temp[:temp.index('_')]
def f(full, ps):
	idx = np.zeros((0,))
	for p in ps:
		idx = np.hstack((idx, np.where(full==p)[0]))
	return idx.astype('int')

def get_indices(people, all_people):
	np.random.shuffle(people)
	train_people = people[0:16]
	test_people = people[16:]
	train_idx = f(all_people, train_people)
	test_idx = f(all_people, test_people)
	return train_idx, test_idx
	
X = np.load(open('X_kth.npy'))
Y = np.load(open('Y_kth.npy'))

files = glob('../Datasets/KTH/Raw Videos/videos/*.npy')
all_people = np.array([get_person(e) for e in files])
people = np.unique(all_people)
train_idx, test_idx = get_indices(people, all_people)
xtrain = X[train_idx, :]
ytrain = Y[train_idx]
xtest = X[test_idx, :]
ytest = Y[test_idx]


svm = SVC(kernel='linear').fit(xtrain, ytrain)
print svm.score(xtest, ytest)
