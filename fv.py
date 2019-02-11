from glob import glob
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np

def powernorm(x):
    return np.sign(x) * np.sqrt(np.abs(x))

def l2norm(x):
    return x / np.linalg.norm(x)

def fv(xx, gmm):
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    fv = np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
    
    fv = powernorm(fv)
    
    fv = l2norm(fv)
    
    return fv

def rootsift(x):
    for i, e in enumerate(x):
        e = e / np.linalg.norm(e, ord=1)
        e = np.sqrt(e)
        x[i] = e
    return x

def norm_hist_feats(x):
    traj = x[:, :10]
    hog = rootsift(x[:, 10:116])
    hof = rootsift(x[:, 116:214])
    mbhx = rootsift(x[:, 214:310])
    mbhy = rootsift(x[:, 310:])
    
    pca_traj = PCA(n_components=5).fit(traj)
    pca_hog = PCA(n_components=48).fit(hog)
    pca_hof = PCA(n_components=54).fit(hof)
    pca_mbhx = PCA(n_components=48).fit(mbhx)
    pca_mbhy = PCA(n_components=48).fit(mbhy)
    
    joblib.dump(pca_traj, 'pca_traj.sav')
    joblib.dump(pca_hog, 'pca_hog.sav')
    joblib.dump(pca_hof, 'pca_hof.sav')
    joblib.dump(pca_mbhx, 'pca_mbhx.sav')
    joblib.dump(pca_mbhy, 'pca_mbhy.sav')
    
    traj = pca_traj.transform(traj)
    hog = pca_hog.transform(hog)
    hof = pca_hof.transform(hof)
    mbhx = pca_mbhx.transform(mbhx)
    mbhy = pca_mbhy.transform(mbhy)
    
    return traj, hog, hof, mbhx, mbhy

def get_blob(m):
    X = []
    for i, (k, v) in enumerate(m.iteritems()):
        print i + 1, len(m.keys())
        for e in v:
            X.append(e)
    X = np.array(X)
    np.random.shuffle(X)
    return X[:256000]

def train_gmm(x):
    traj, hog, hof, mbhx, mbhy = norm_hist_feats(x)
    
    traj_gmm = GMM(n_components=256).fit(traj)
    joblib.dump(traj_gmm, 'traj_gmm.sav')
    
    hog_gmm = GMM(n_components=256).fit(hog)
    joblib.dump(hog_gmm, 'hog_gmm.sav')
    
    hof_gmm = GMM(n_components=256).fit(hof)
    joblib.dump(hof_gmm, 'hof_gmm.sav')
    
    mbhx_gmm = GMM(n_components=256).fit(mbhx)
    joblib.dump(mbhx_gmm, 'mbhx_gmm.sav')
    
    mbhy_gmm = GMM(n_components=256).fit(mbhy)
    joblib.dump(mbhy_gmm, 'mbhy_gmm.sav')

def compute_fvs(dm, train=True):
    gmm_traj = joblib.load('traj_gmm.sav')
    gmm_hog = joblib.load('hog_gmm.sav')
    gmm_hof = joblib.load('hof_gmm.sav')
    gmm_mbhx = joblib.load('mbhx_gmm.sav')
    gmm_mbhy = joblib.load('mbhy_gmm.sav')
    
    pca_traj = joblib.load('pca_traj.sav')
    pca_hog = joblib.load('pca_hog.sav')
    pca_hof = joblib.load('pca_hof.sav')
    pca_mbhx = joblib.load('pca_mbhx.sav')
    pca_mbhy = joblib.load('pca_mbhy.sav')
    
    for k, x in dm.iteritems():
        traj = x[:, :10]
        hog = rootsift(x[:, 10:116])
        hof = rootsift(x[:, 116:214])
        mbhx = rootsift(x[:, 214:310])
        mbhy = rootsift(x[:, 310:])
        
        traj = pca_traj.transform(traj)
        hog = pca_hog.transform(hog)
        hof = pca_hof.transform(hof)
        mbhx = pca_mbhx.transform(mbhx)
        mbhy = pca_mbhy.transform(mbhy)
        
        traj_fv = l2norm(fv(traj, gmm_traj))
        hog_fv = l2norm(fv(hog, gmm_hog))
        hof_fv = l2norm(fv(hof, gmm_hof))
        mbhx_fv = l2norm(fv(mbhx, gmm_mbhx))
        mbhy_fv = l2norm(fv(mbhy, gmm_mbhy))
        
        full_fv = np.hstack((traj_fv, hog_fv, hof_fv, mbhx_fv, mbhy_fv))
        
        if train:
            np.save('fvs/train_{}'.format(k.split('/')[-1]), full_fv)
        else:
            np.save('fvs/test_{}'.format(k.split('/')[-1]), full_fv)
        
def load_data(fs):
    m = {}
    for i, f in enumerate(fs):
        print i + 1, len(fs)

        a = open(f).read().splitlines()
        x = []
        for e in a:
            x.append(np.array(e.split('\t')[:-1]).astype('float64'))
        x = np.array(x)
        m[f] = x
    return m

train_ppl = list(np.load('train_ppl.npy'))
test_ppl = list(np.load('test_ppl.npy'))

files = glob('idt/*.txt')
train_files = [e for e in files if e.split('/')[-1].split('_')[0] in train_ppl]
test_files = [e for e in files if e.split('/')[-1].split('_')[0] in test_ppl]

print len(train_files), len(test_files)

tr_data_map = load_data(train_files)
te_data_map = load_data(test_files)


#X_tr = get_blob(tr_data_map)
#print X_tr.shape
#print 'Training GMM'
#train_gmm(X_tr)

compute_fvs(tr_data_map, train=True)
compute_fvs(te_data_map, train=False)
