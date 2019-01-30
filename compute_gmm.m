function [means, covariances, priors] = compute_gmm(X, K)

    disp('Compute GMM');
    X = sample_rows(X, 256000);
    [means, covariances, priors] = vl_gmm(X', K);

end
