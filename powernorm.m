function [v] = powernorm(v)

    v = sign(v) .* sqrt(abs(v));

end
