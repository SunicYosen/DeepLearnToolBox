%%=========================================================================
%函数名称:cnnff（）
%输入参数:net，神经网络；x，训练数据矩阵；
%输出参数:net，训练完成的卷积神经网络
%主要功能:使用当前的神经网络对输入的向量进行预测
%算法流程:1）将样本打乱，随机选择进行训练；
%        2）讲样本输入网络，层层映射得到预测值
%注意事项:1）使用BP算法计算梯度
%%=========================================================================
% 完成训练前向过程

function net = cnnff(net, x)
    n = numel(net.layers);    %层数
    net.layers{1}.a{1} = x;   %网络的第一层就是输入，但这里的输入包含了多个训练图像
    inputmaps = 1;            %输入层只有一个特征map，也就是原始的输入图像

    for l = 2 : n   %  for each layer 对于每层(第一层是输入层，循环时先忽略掉）
        if strcmp(net.layers{l}.type, 'c') %如果当前是卷积层
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map 对每一个输出map，需要用outputmaps个不同的卷积核去卷积图像
                %  create temp output map
                %%=========================================================================
                %主要功能：创建outmap的中间变量，即特征矩阵
                %实现步骤：用这个公式生成一个零矩阵，作为特征map
                %注意事项：1）对于上一层的每一张特征map，卷积后的特征map的大小是：（输入map宽 - 卷积核的宽 + 1）* （输入map高 - 卷积核高 + 1）
                %         2）由于每层都包含多张特征map，则对应的索引则保存在每层map的第三维，及变量Z中
                %%=========================================================================

                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                
                for i = 1 : inputmaps   % for each input map 对于输入的每个特征map
                    %%=========================================================================
                    %主要功能：将上一层的每一个特征map（也就是这层的输入map）与该层的卷积核进行卷积
                    %实现步骤：1）进行卷积
                    %         2）加上对应位置的基b，然后再用sigmoid函数算出特征map中每个位置的激活值，作为该层输出特征map
                    %注意事项：1）当前层的一张特征map，是用一种卷积核去卷积上一层中所有的特征map，然后所有特征map对应位置的卷积值的和
                    %         2）有些论文或者实际应用中，并不是与全部的特征map链接的，有可能只与其中的某几个连接
                    %%=========================================================================

                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j}); %加上偏置b
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;   %更新当前层的输入map的数量

        elseif strcmp(net.layers{l}.type, 's')      %下采样层
            %%=========================================================================
            %主要功能：对特征map进行下采样
            %实现步骤：1）进行卷积
            %         2）最终pooling的结果需要从上面得到的卷积结果中以scale=2为步长，跳着把mean pooling的值读出来
            %注意事项：1）例如我们要在scale=2的域上面执行mean pooling，那么可以卷积大小为2*2，每个元素都是1/4的卷积核
            %         2）因为convn函数的默认卷积步长为1，而pooling操作的域是没有重叠的，所以对于上面的卷积结果
            %         3）是利用卷积的方法实现下采样
            %%=========================================================================
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :); %跳读mean pooling的值
            end
        end
    end

    %%=========================================================================
    %主要功能：输出层，将最后一层得到的特征变成一条向量，作为最终提取得到的特征向量
    %实现步骤：1）获取倒数第二层中每个特征map的尺寸
    %         2）用reshape函数将map转换为向量的形式
    %         3）使用sigmoid(W*X + b)函数计算样本输出值，放到net成员o中
    %注意事项：1）在使用sigmoid（）函数是，是同时计算了batchsize个样本的输出值
    %%=========================================================================

    %  concatenate all end layer feature maps into vector
    net.fv = [];            %net.fv为神经网络倒数第二层的输出map
    for j = 1 : numel(net.layers{n}.a)  %最后一层的特征map的个数
        sa = size(net.layers{n}.a{j});  %第j个特征map的大小
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));  %通过全连接层的映射得到网络的最终预测结果输出
end
