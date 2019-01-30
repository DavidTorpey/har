function [X_s] = sample_rows(X, p)

    idx = randperm(length(X));
    m = ceil(p * length(X));
    X_s = X(idx(1:m), :);

end
