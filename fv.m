run('/home-cuda/dtorpey/vlfeat-0.9.21/toolbox/vl_setup');

files = dir('../iDT/*.txt');
fns = cell(1,0);
N = 150;
for i = 1:N
    fns{i} = files(i).name;
end

for i = 1:N
    test_idx = i;
    train_idx = 1:N;
    train_idx(i) = [];
    
    test_files = fns(test_idx);
    train_files = fns(train_idx);
    
    [X_traj, X_hog, X_hof, X_mbhx, X_mbhy] = gen_feat_mat(i, train_files);
    
    [traj_means, traj_covariances, traj_priors] = compute_gmm(X_traj, 256);
    [hog_means, hog_covariances, hog_priors] = compute_gmm(X_hog, 256);
    [hof_means, hof_covariances, hof_priors] = compute_gmm(X_hof, 256);
    [mbhx_means, mbhx_covariances, mbhx_priors] = compute_gmm(X_mbhx, 256);
    [mbhy_means, mbhy_covariances, mbhy_priors] = compute_gmm(X_mbhy, 256);
    
    compute_fv(i, train_files, 'train', traj_means, traj_covariances, traj_priors, hog_means, hog_covariances, hog_priors, hof_means, hof_covariances, hof_priors, mbhx_means, mbhx_covariances, mbhx_priors, mbhy_means, mbhy_covariances, mbhy_priors);

    compute_fv(i, test_files, 'test', traj_means, traj_covariances, traj_priors, hog_means, hog_covariances, hog_priors, hof_means, hof_covariances, hof_priors, mbhx_means, mbhx_covariances, mbhx_priors, mbhy_means, mbhy_covariances, mbhy_priors);
    
end
