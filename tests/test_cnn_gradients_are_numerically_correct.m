function test_cnn_gradients_are_numerically_correct
fprintf('Enter test_cnn_gradients_are_numerically_correct ...\n')
batch_x = rand(28,28,5);    %28 x 28 x 5 数组
batch_y = rand(10,5);       %10 x 5 矩阵 

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, batch_x, batch_y);

cnn = cnnff(cnn, batch_x);
cnn = cnnbp(cnn, batch_y);
cnnnumgradcheck(cnn, batch_x, batch_y);