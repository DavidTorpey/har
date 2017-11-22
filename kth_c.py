import numpy as np
from glob import glob
from keras.applications import ResNet50
import cv2

def get_action(f):
	actions = ['walking', 'jogging', 'running', 'boxing', 'waving', 'clapping']
	for action in actions:
		if action.lower() in f.lower():
			return action

def get_spatial(video):
        spatial_features = np.zeros((video.shape[-1], 224, 224, 3))
        for i in range(video.shape[-1]):
		frame = video[:, :, i]
                spatial_features[i, :, :, :] = cv2.resize(np.stack((frame, frame, frame), axis=2), s)
        return resnet.predict(spatial_features).mean(0).ravel()

def get_temporal(video, T):
        temporal_features = np.zeros((video.shape[-1] - T, 224, 224, 3))
        for i in range(video.shape[-1] - T):
                frame0 = cv2.resize(np.stack((video[:, :, i], video[:, :, i], video[:, :, i]), axis=2), s).astype('float64')
                frame1 = cv2.resize(np.stack((video[:, :, i + T], video[:, :, i + T], video[:, :,  i + T]), axis=2), s).astype('float64')
                frame = frame1 - frame0
                temporal_features[i, :, :, :] = frame
        return resnet.predict(temporal_features).mean(0).ravel()

def vectorise_video(video):
        v_video = np.zeros((video.shape[0] * video.shape[1], video.shape[2]))
        for i in range(video.shape[-1]):
                v_video[:, i] = video[:, :, i].ravel()
        return v_video

resnet = ResNet50(include_top=False)
s = (224, 224)

files = glob('../Datasets/KTH/Raw Videos/videos/*.npy')
B = 200
X = np.zeros((len(files), 4096 + B))
K = 80
T = 1
for i, f in enumerate(files):
	print i
	video = np.load(open(f))['array1']
	video_sampled = video[:, :, np.round(np.linspace(0, video.shape[-1] - 1, K)).astype('int')]

	vid_cov = np.cov(vectorise_video(video_sampled).T)
        cov_feat = np.histogram(vid_cov.ravel(), bins=B)[0]

	spatial_feature = get_spatial(video_sampled)
	temporal_feature = get_temporal(video_sampled, T)
	video_feature = np.hstack((spatial_feature, temporal_feature, cov_feat))
	X[i, :] = video_feature
Y = np.array([get_action(e) for e in files])



np.save(open('X_kth.npy', 'wb'), X)
np.save(open('Y_kth.npy', 'wb'), Y)
