clear all

for k = 1 : 30
    in(:,:,k) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/p',num2str(k),'.bmp'])); % 파일 가져오기
    out(k,:) = 1;
end

for k = 1 : 30
    in(:,:,k+30) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/m',num2str(k),'.bmp'])); % 파일 가져오기
    out(k+30,:) = 2;
end

for k = 1 : 30
    in(:,:,k+60) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/x',num2str(k),'.bmp'])); % 파일 가져오기
    out(k+60,:) = 3;
end

idx =randperm(60,60);
X = in(:,:,idx);
D = out(idx,:);

% Learning
%
W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1);
Wo = (2*rand( 3,  100) - 1);

for epoch = 1:10
    epoch
  [W1, W5, Wo] = MnistConv2(W1, W5, Wo, X, D);
end

save('MnistConv5.mat');


% Test
%
for k = 1 : 30
    X(:,:,k) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/p',num2str(k),'.bmp'])); % 파일 가져오기
    D(k,:) = 1;
end

for k = 1 : 30
    X(:,:,k+30) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/m',num2str(k),'.bmp'])); % 파일 가져오기
    D(k+30,:) = 2;
end

for k = 1 : 30
    X(:,:,k+60) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/data/x',num2str(k),'.bmp'])); % 파일 가져오기
    D(k+60,:) = 3;
end

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
  v(isnan(v)) = 0;
  y  = Softmax(v);                  %
  y(isnan(y))=0;

  [~, i] = max(y);
  if i == D(k)
    acc = acc + 1;
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);


