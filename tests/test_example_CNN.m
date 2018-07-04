function test_example_CNN
fprintf('Enter test_example_CNN ...\n');

load mnist_uint8;
%% 该模型采用的数据为mnist_uint8.mat，含有70000个手写数字样本其中60000作为训练样本，10000作为测试样本。 
%% 把数据转成相应的格式，并归一化。

train_x = double(reshape(train_x',28,28,60000))/255;
%% train_x' 矩阵转置,训练样本 [train_x FROM mnist_unint8] 

test_x = double(reshape(test_x',28,28,10000))/255; 
%测试样本 [mnist_uint8 样本] [test_x FROM mnist_unint8] 

train_y = double(train_y'); %[train_y FROM mnist_unint8] 
test_y = double(test_y');   %[test_y FROM mnist_unint8] 

% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
% will run 1 epoch in about 200 second and get around 11% error. 
% With 100 epochs you'll get around 1.2% error

rand('state',0) %

cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
};

%% 训练选项，alpha学习效率（不用），batchsiaze批训练总样本的数量，numepoches迭代次数
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

% 初始化网络，对数据进行批训练，验证模型准确率
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
% 绘制均方差曲线
figure; plot(cnn.rL);

assert(er<0.12, 'Too big error');
