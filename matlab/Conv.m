function y = Conv(x, W)

[wrow, wcol, numFilters] = size(W);     % 9x9x20
[xrow, xcol, ~         ] = size(x);     % 28x28

yrow = xrow - wrow + 1;                 % 28 - 9 + 1 = 20
ycol = xcol - wcol + 1;                 % 28 - 9 + 1 = 20

y = zeros(yrow, ycol, numFilters);      % 20x20x20

for k = 1:numFilters
  filter = W(:, :, k);                  % 9x9x1
  filter = rot90(squeeze(filter), 2);   % 180도 회전
  y(:, :, k) = conv2(x, filter, 'valid'); 
end

end
