function [X_s] = sample_rows(X, p)

    idx = randperm(length(X));
    if p < 1
        m = ceil(p * length(X));
    else
	m = p;
    end

    X_s = X(idx(1:m), :);

end
