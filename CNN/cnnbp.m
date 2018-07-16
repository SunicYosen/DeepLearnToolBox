%%=========================================================================================================
%函数名称：cnnbp（）
%输入参数：net，呆训练的神经网络；y，训练样本的标签，即期望输出
%输出参数：net，经过BP算法训练得到的神经网络
%主要功能：通过BP算法训练神经网络参数
%实现步骤：1）将输出的残差扩展成与最后一层的特征map相同的尺寸形式
%         2）如果是卷积层，则进行上采样
%         3）如果是下采样层，则进行下采样
%         4）采用误差传递公式对灵敏度进行反向传递
%注意事项：1）从最后一层的error倒推回来deltas，和神经网络的BP十分相似，可以参考“UFLDL的反向传导算法”的说明
%         2）在fvd里面保存的是所有样本的特征向量（在cnnff.m函数中用特征map拉成的），所以这里需要重新换回来特征map的形式，
%            d保存的是delta，也就是灵敏度或者残差
%         3）net.o .* (1 - net.o))代表输出层附加的非线性函数的导数，即sigm函数的导数
%%=========================================================================================================

%计算并传递神经网络的Error，并计算梯度（权重的修改量）
function net = cnnbp(net, y)
    n = numel(net.layers); %网络层数

    %   error
    net.e = net.o - y; %实际输出入期望输出的差值
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);   %代价函数，采用均方差函数作为代价函数

    %%  backprop deltas
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta  % 输出层的灵敏度或者残差,(net.o .* (1 - net.o))代表输出层的激活函数的导数
    net.fvd = (net.ffW' * net.od);              %  feature vector delta %残差反向传播回前一层，net.fvd保存的是残差
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function 只有卷积层采用sigm函数
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));  %net.fv是前一层的输出（未经过simg函数），作为输出层的输入
    end

    %%%%%%%%%%%%%%%%%%%%将输出的残差扩展成与最后一层的特征map相同的尺寸形式%%%%%%%%%%%%%%%%%%%%
    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
