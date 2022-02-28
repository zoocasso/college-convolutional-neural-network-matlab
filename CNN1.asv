clear all

Images = loadMNISTImages('/Users/zoocasso/Documents/MATLAB/t10k-images.idx3-ubyte'); % 파일 가져오기
Images = reshape (Images, 28, 28, []); % 28 x 28 x 10000
Labels = loadMNISTLabels('/Users/zoocasso/Documents/MATLAB/t10k-labels.idx1-ubyte');% 파일 가져오기 // 0~9
Labels(Labels == 0) = 10; 

rng(1);

% Learning
%
W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1);
Wo = (2*rand( 10,  100) - 1);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:10
  epoch
  [W1, W5, Wo] = MnistConv1(W1, W5, Wo, X, D);
end

save('MnistConv1.mat');


% Test
%
X = Images(:, :, 8001:10000);
D = Labels(8001:10000);

acc = 0;
N   = length(D);
for k = 1:N
  x = X(:, :, k);                   % Input,           28x28

  y1 = Conv(x, W1);                 % Convolution,  20x20x20
  y2 = ReLU(y1);                    %
  y3 = Pool(y2);                    % Pool,         10x10x20
  y4 = reshape(y3, [], 1);          %                   2000  
  v5 = W5*y4;                       % ReLU,              360
  y5 = ReLU(v5);                    %
  v  = Wo*y5;                       % Softmax,            10
  y  = Softmax(v);                  %

  [~, i] = max(y);
  if i == D(k)
    acc = acc + 1;
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);


