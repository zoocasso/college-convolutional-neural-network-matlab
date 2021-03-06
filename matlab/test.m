%CNN1
load("MnistConv1.mat");
%input image
X(:,:,1) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/test/test_2.bmp']));
x = X(:, :, 1);                   % Input,           28x28

  y1 = Conv(x, W1);                 % Convolution,  20x20x20
  y2 = ReLU(y1);                    %
  y3 = Pool(y2);                    % Pool,         10x10x20
  y4 = reshape(y3, [], 1);          %                   2000  
  v5 = W5*y4;                       % ReLU,              360
  y5 = ReLU(v5);                    %
  v  = Wo*y5;                       % Softmax,            10
  y  = Softmax(v);                  %
  cnn1 = [y]';


%CNN2
load('MnistConv2.mat');
%input image
X(:,:,1) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/test/test_5.bmp']));
x = X(:, :, 1);                   % Input,           28x28

  y1 = Conv(x, W1);                 % Convolution,  20x20x20
  y2 = ReLU(y1);                    %
  y3 = Pool(y2);                    % Pool,         10x10x20
  y4 = reshape(y3, [], 1);          %                   2000  
  v5 = W5*y4;                       % ReLU,              360
  y5 = ReLU(v5);                    %
  v  = Wo*y5;                       % Softmax,            10
  y  = Softmax(v);                  %
  cnn2 = [y]';

%CNN3
load('MnistConv3.mat');
%input image
X(:,:,1) = im2double(imread(['/Users/zoocasso/Documents/MATLAB/test/test_m.bmp']));
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
  cnn3 = [y]';

%one hot encoding -> 8bit
load('NN.mat');
%input 13bit
NNinput = horzcat(cnn1,cnn2,cnn3);
x = NNinput(1,:)';
        v1 = W1*x;
        y1 = Sigmoid(v1);
        
        %2nd Hidden Layer
        v2 = W2*y1;
        y2 = Sigmoid(v2);
        
        %3rd Hidden Layer
        v3 = W3*y2;
        y3 = Sigmoid(v3);
        
        %Output Layer
        v = W4*y3;
        y = Sigmoid(v)';
        eightBit= y;

%7segment
load('sevenSegment.mat');
%input 8bit
%1st Hidden Layer
        x = eightBit';
        v1 = W1*x;
        y1 = Sigmoid(v1);
        
        %2nd Hidden Layer
        v2 = W2*y1;
        y2 = Sigmoid(v2);
        
        %3rd Hidden Layer
        v3 = W3*y2;
        y3 = Sigmoid(v3);
        
        %Output Layer
        v = W4*y3;
        y = Sigmoid(v)'