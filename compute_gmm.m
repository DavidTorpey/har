function [means, covariances, priors] = compute_gmm(X, K)

    disp('Compute GMM');
    [means, covariances, priors] = vl_gmm(X', K);

end
