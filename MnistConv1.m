function [W1, W5, Wo] = MnistConv1(W1, W5, Wo, X, D)
%
%
    
    alpha = 0.01;   %Learning rate
    beta  = 0.95;   %momentum
    
    momentum1 = zeros(size(W1));
    momentum5 = zeros(size(W5));
    momentumo = zeros(size(Wo));
    
    N = length(D);  % N = 8000
    
    bsize = 100;    %batch size
    blist = 1:bsize:(N-bsize+1);    %1:100:8000-100+1 -> 1:100:7901 (1,101,201, ... , 7901)
    
    % One epoch loop
    %
    for batch = 1:length(blist)     % 1 ~ 80
      dW1 = zeros(size(W1));
      dW5 = zeros(size(W5));
      dWo = zeros(size(Wo));
      
      % Mini-batch loop
      %
      begin = blist(batch);         % 1
      for k = begin:begin+bsize-1   % 1: 1+100-1 -> 1:100
        % Forward pass = inference
        %
        x  = X(:, :, k);               % Input,           28x28
        y1 = Conv(x, W1);              % Convolution,  20x20x20
        y2 = ReLU(y1);                 % 20x20x20
        y3 = Pool(y2);                 % Pooling,      10x10x20
        y4 = reshape(y3, [], 1);       % 2000x1
        v5 = W5*y4;                    % ReLU,         (100x2000) x (2000x1) -> 100x1
        y5 = ReLU(v5);                 % 100x1
        v  = Wo*y5;                    % Softmax,      (10x100) x (100x1) -> 10x1
        v(isnan(v)) = 0;
        y  = Softmax(v);               % 10x1
        y(isnan(y)) = 0;
    
        % One-hot encoding
        %
        d = zeros(10, 1);              % 10x1
        d(sub2ind(size(d), D(k), 1)) = 1;   % k번째 위치에 1을 설정
    
        % Backpropagation
        %
        e      = d - y;                   % Output layer  10x1
        delta  = e;                         %Cross Entropy 10x1
    
        e5     = Wo' * delta;             % Hidden(ReLU) layer (100x10) x (10x1) -> (100x1)
        delta5 = (y5 > 0) .* e5;            %100x1
    
        e4     = W5' * delta5;            % Pooling layer 100x1
        
        e3     = reshape(e4, size(y3));     % 10x10x20
    
        e2 = zeros(size(y2));           
        W3 = ones(size(y2)) / (2*2);
        for c = 1:20
          e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
        end
        
        delta2 = (y2 > 0) .* e2;          % ReLU layer
      
        delta1_x = zeros(size(W1));       % Convolutional layer
        for c = 1:20
          delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
        end
        
        dW1 = dW1 + delta1_x; 
        dW5 = dW5 + delta5*y4';    
        dWo = dWo + delta *y5';
      end 
      
      % Update weights
      %
      dW1 = dW1 / bsize;
      dW5 = dW5 / bsize;
      dWo = dWo / bsize;
      
      momentum1 = alpha*dW1 + beta*momentum1;
      W1        = W1 + momentum1;
      
      momentum5 = alpha*dW5 + beta*momentum5;
      W5        = W5 + momentum5;
       
      momentumo = alpha*dWo + beta*momentumo;
      Wo        = Wo + momentumo;  
    end

end

