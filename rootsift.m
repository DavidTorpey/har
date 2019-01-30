function [a] = rootsift(a)

    for i = 1:size(a,1)
        a(i, :) = a(i, :) / norm(a(i, :), 1);
    end
    a = sign(a) .* sqrt(abs(a));

end

