function y = MaxPool(x)
%     
% 2x2 max pooling
%
%
[xrow, xcol, numFilters] = size(x);     %20x20x20

y = zeros(xrow/2, xcol/2, numFilters);  %10x10x20
ym = [];
ym1 =[];
for k = 1:numFilters                    %1~20
    for m = 1:2:xrow-1                  %1:2:20-1 -> 1:2:19 (1, 3, 5, ... , 19)
        for n = 1:2:xcol-1              %1:2:20-1 -> 1:2:19 (1, 3, 5, ... , 19)
            max_y = max(max(max(x(m:m+1, n:n+1, k))));
            ym = [ym, max_y(1)];
        end
        ym1 = [ym1; ym];
        ym =[];
    end
    y(:,:,k) = ym1;
    ym1=[];
end
end
 