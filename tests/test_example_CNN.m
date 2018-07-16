%%=========================================================================
% 主要功能：在mnist数据库上做实验，验证工具箱的有效性
% 算法流程：1）载入训练样本和测试样本
%          2）设置CNN参数，并进行训练
%          3）进行检测cnntest()
% 注意事项：1）由于直接将所有测试样本输入会导致内存溢出，故采用一次只测试一个训练样本的测试方法
%%=========================================================================

function test_example_CNN
fprintf('Enter test_example_CNN ...\n');

load('/home/sun/Files/Matlab/DeepLearnToolbox/data/imgLabel.mat');
load('/home/sun/Files/Matlab/DeepLearnToolbox/data/imgArrayPack.mat');
%load mnist_uint8;
%% 该模型采用的数据为mnist_uint8.mat，含有70000个手写数字样本其中60000作为训练样本，10000作为测试样本。 
%% 把数据转成相应的格式，并归一化。

imgReshape = reshape(imgArrayPack',64,64,3400);
%
%for img=1:3400
%    image = imgReshape(:,:,img);
%    imshow(image);
%    fprintf('%d\n',img);
%end
%
train_x = double(imgReshape)/255;
%train_x = double(reshape(train_x',28,28,60000))/255;
%% train_x' 矩阵转置,训练样本 [train_x FROM mnist_unint8]  图片数据集合

test_x = double(imgReshape)/255;
%test_x = double(reshape(test_x',28,28,10000))/255; 
%测试样本 [mnist_uint8 样本] [test_x FROM mnist_unint8]   测试图片数据集

train_y = double(imgLabel');
test_y = double(imgLabel');
%train_y = double(train_y');     %[train_y FROM mnist_unint8]  图片的标签集合
%test_y = double(test_y');        %[test_y FROM mnist_unint8]  测试的标签

% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
% will run 1 epoch in about 200 second and get around 11% error. 
% With 100 epochs you'll get around 1.2% error
rand('state',0) %

% 设置CNN的基本参数规格，如卷积、降采样层的数量，卷积核的大小、降采样的降幅
cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize',9)   %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize',9)  %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
};

%% 训练选项，alpha学习效率（不用），batchsiaze批训练总样本的数量，numepoches迭代次数
opts.alpha = 1;
opts.batchsize = 5;
opts.numepochs =1;
opts.stepsize = 1;

cnn.activation = 'Relu';            %Relu or Sigmoid Activation 
%cnn.pooling_mode = 'Max';    %Mean and Max Pooling
%cnn.output = 'Softmax';           %

% 初始化网络，对数据进行批训练，验证模型准确率
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);
% plot mean squared error
% 绘制均方差曲线
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
