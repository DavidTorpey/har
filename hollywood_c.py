import numpy as np
from glob import glob
import cv2
from keras.applications import ResNet50

def get_action(f):
	actions = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake', 'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']	
	for action in actions:
		if action.lower() in f.lower():
			return action

def get_spatial(video):
        spatial_features = np.zeros((video.shape[0], 224, 224, 3))
        for i in range(video.shape[0]):
                spatial_features[i, :, :, :] = cv2.resize(video[i, :, :, :], s)
        return resnet.predict(spatial_features).mean(0).ravel()

def get_temporal(video, T):
        temporal_features = np.zeros((video.shape[0] - T, 224, 224, 3))
        for i in range(video.shape[0] - T):
                frame0 = cv2.resize(video[i, :, :, :], s).astype('float64')
                frame1 = cv2.resize(video[i + T, :, :, :], s).astype('float64')
                frame = frame1 - frame0
                temporal_features[i, :, :, :] = frame
        return resnet.predict(temporal_features).mean(0).ravel()

def vectorise_video(video):
        v_video = np.zeros((video.shape[1] * video.shape[2] * video.shape[3], video.shape[0]))
        for i in range(video.shape[0]):
                v_video[:, i] = video[i, :, :, :].ravel()
        return v_video

resnet = ResNet50(include_top=False)
s = (224, 224)

actions = glob('../Datasets/Hollywood2/Hollywood2/Files/*')
B = 200
xtrain = np.zeros((0, 4096 + B))
ytrain = []
xtest = np.zeros((0, 4096 + B))
ytest = []
T = 1
K = 500
c = 0
for action in actions:
	files = glob(action + '/*.npy')
	for f in files:
		try:
			video = np.load(open(f))['array1']
			video_sampled = video[np.round(np.linspace(0, video.shape[0] - 1, K)).astype('int'), :, :, :]
			spatial_feature = get_spatial(video_sampled)
			temporal_feature = get_temporal(video_sampled, T)

			vid_cov = np.cov(vectorise_video(video_sampled).T)
		        cov_feat = np.histogram(vid_cov.ravel(), bins=B)[0]

			video_feature = np.hstack((spatial_feature, temporal_feature, cov_feat))
			a = get_action(f)
			if 'train' in f:
				xtrain = np.vstack((xtrain, video_feature))
				ytrain.append(a)
			else:
				xtest = np.vstack((xtest, video_feature))
				ytest.append(a)
		except Exception as e:
			print e
		print c
		c = c + 1

np.save(open('hollywood_Xtr.npy', 'wb'), xtrain)
np.save(open('hollywood_Ytr.npy', 'wb'), ytrain)
np.save(open('hollywood_Xte.npy', 'wb'), xtest)
np.save(open('hollywood_Yte.npy', 'wb'), ytest)

