function [proj] = perform_pca(X, pcs)

    coeff = pca(X);
    coeff = coeff(:, 1:pcs);
    proj = X * coeff;

end

