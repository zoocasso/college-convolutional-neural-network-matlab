function [W1, W2, W3, W4] = BackpropCE1(W1, W2, W3, W4, X, D)
  alpha = 0.2;          %Learning Rate
  N = 300;               %총 300개
  for k = 1:N
      %Input Set {input data, Correct Answer}
    x = X(k, :)';        % x = a column vector
    d = D(k, :)';
    
    %1st Hidden Layer
    v1 = W1*x;          %20x1 = 20x23 * 23x1
    y1 = Sigmoid(v1);   %20x1

    %2nd Hidden Layer
    v2 = W2*y1;         %20x1 = 20x20 * 20x1
    y2 = Sigmoid(v2);   %20x1

    %3rd Hidden Layer
    v3 = W3*y2;         %20x1 = 20x20 * 20x1
    y3 = Sigmoid(v3);   %20x1

    %Output Layer
    v  = W4*y3;         %8x1 = 8x20 * 20x1
    y  = Sigmoid(v);    %8x1
    
    %Back-propagation
    e     = d - y;      
    delta = e;

    e3     = W4'*delta;
    delta3 = y3.*(1-y3).*e3;

    e2     = W3'*delta3;
    delta2 = y2.*(1-y2).*e2;

    e1     = W2'*delta2;
    delta1 = y1.*(1-y1).*e1;
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;

    dW2 = alpha*delta2*y1';    
    W2 = W2 + dW2;

    dW3 = alpha*delta3*y2';
    W3 = W3 + dW3;

    dW4 = alpha*delta*y3';
    W4 = W4 + dW4;
  end
  
end