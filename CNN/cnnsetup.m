% 该函数你用于初始化CNN的参数。 
% 设置各层的mapsize大小， 
% 初始化卷积层的卷积核、bias 
% 尾部单层感知机的参数设置

%%=============================================================================================
% 函数名称：cnnsetup
% 输入参数：net，待设置的卷积神经网络；x，训练样本；y，训练样本对应标签；
% 输出参数：net，初始化完成的卷积神经网络
% 主要功能：对CNN的结构进行初始化
% 算法流程：1）
% 注意事项：1）isOctave这个语句是为了抛出一个程序在Octave平台上运行时的一个BUG，在matlab平台上可以直接注释掉
%          2）net.layers中有五个struct类型的元素，实际上就表示CNN共有五层，这里范围的是5
%%=============================================================================================

function net = cnnsetup(net, x, y)
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;  %初始化网络的输入层为1层

    %%=========================================================================
    % 主要功能：得到输入图像的行数和列数
    % 注意事项：1）B=squeeze(A) 返回和矩阵A相同元素但所有单一维都移除的矩阵B，单一维是满足size(A,dim)=1的维。
    %             train_x中图像的存放方式是三维的reshape(train_x',28,28,60000)，前面两维表示图像的行与列，
    %             第三维就表示有多少个图像。这样squeeze(x(:, :, 1))就相当于取第一个图像样本后，再把第三维
    %             移除，就变成了28x28的矩阵，也就是得到一幅图像，再size一下就得到了训练样本图像的行数与列数了
    %%=========================================================================

    mapsize = size(squeeze(x(:, :, 1))); %[28, 28];一个行向量。x(:, :, 1)是一个训练样本。

    % 下面通过传入net这个结构体来逐层构建CNN网络
    for l = 1 : numel(net.layers)           % Layer 对于每一层
        if strcmp(net.layers{l}.type, 's')  % 降采样层sub sampling 对于当前层是采样层

            %%=========================================================================
            % 主要功能：获取下采样之后特征map的尺寸
            % 注意事项：1）subsampling层的mapsize，最开始mapsize是每张图的大小28*28
            %             这里除以scale=2，就是pooling之后图的大小，pooling域之间没有重叠，所以pooling后的图像为14*14
            %             注意这里的右边的mapsize保存的都是上一层每张特征map的大小，它会随着循环进行不断更新
            %%=========================================================================

            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            
            for j = 1 : inputmaps           % 一个降采样层的所有输入map，b初始化为0 | 对于上一层的每一个特征图
                net.layers{l}.b{j} = 0;     % 对偏执初始化为零
            end
        end

        if strcmp(net.layers{l}.type, 'c')  %如果当前层是卷积层
            
            %%=========================================================================
            % 主要功能：获取卷积后的特征map尺寸以及当前层待学习的卷积核的参数数量
            % 注意事项：1）旧的mapsize保存的是上一层的特征map的大小，那么如果卷积核的移动步长是1，那用
            %             kernelsize * kernelsize大小的卷积核卷积上一层的特征map后，得到的新的map的大小就是下面这样
            %          2）fan_out代表该层需要学习的参数个数。每张特征map是一个(后层特征图数量)*(用来卷积的patch图的大小)
            %             因为是通过用一个核窗口在上一个特征map层中移动（核窗口每次移动1个像素），遍历上一个特征map
            %             层的每个神经元。核窗口由kernelsize*kernelsize个元素组成，每个元素是一个独立的权值，所以
            %             就有kernelsize*kernelsize个需要学习的权值，再加一个偏置值。另外，由于是权值共享，也就是
            %             说同一个特征map层是用同一个具有相同权值元素的kernelsize*kernelsize的核窗口去感受输入上一
            %             个特征map层的每个神经元得到的，所以同一个特征map，它的权值是一样的，共享的，权值只取决于
            %             核窗口。然后，不同的特征map提取输入上一个特征map层不同的特征，所以采用的核窗口不一样，也
            %             就是权值不一样，所以outputmaps个特征map就有（kernelsize*kernelsize+1）* outputmaps那么多的权值了
            %             但这里fan_out只保存卷积核的权值W，偏置b在下面独立保存
            %%=========================================================================

            %得到卷积层的featuremap的size，卷积层fm的大小为: 上一层大小 - 卷积核大小 + 1（步长为1的情况）
            mapsize = mapsize - net.layers{l}.kernelsize + 1;

            %fan_out: 该层的所有连接的数量 = 卷积核数 * 卷积核size = 6*(5*5)，12*(5*5)
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;

            %卷积核初始化，1层卷积为1*6个卷积核，2层卷积一共有6*12=72个卷积核。
            for j = 1 : net.layers{l}.outputmaps  %  output map
                %%=========================================================================
                % 主要功能：获取卷积层与前一层输出map之间需要链接的参数链个数
                % 注意事项：1）fan_out保存的是对于上一层的一张特征map，我在这一层需要对这一张特征map提取outputmaps种特征，
                %             提取每种特征用到的卷积核不同，所以fan_out保存的是这一层输出新的特征需要学习的参数个数
                %             而，fan_in保存的是，我在这一层，要连接到上一层中所有的特征map，然后用fan_out保存的提取特征
                %             的权值来提取他们的特征。也即是对于每一个当前层特征图，有多少个参数链到前层
                %%=========================================================================
                % 输入做卷积

                % fan_in = 本层的一个输出map所对应的所有卷积核，包含的权值的总数 = 1*25,6*25
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;

                for i = 1 : inputmaps  %  input map 对于上一层的每一个输出特征map（本层的输入map)
                    %%=========================================================================
                    % 主要功能：随机初始化卷积核的权值，再将偏置均初始化为零
                    % 注意事项：1）随机初始化权值，也就是共有outputmaps个卷积核，对上层的每个特征map，都需要用这么多个卷积核去卷积提取特征。
                    %             rand(n)是产生n×n的 0-1之间均匀取值的数值的矩阵，再减去0.5就相当于产生-0.5到0.5之间的随机数
                    %             再 *2 就放大到 [-1, 1]。然后再乘以后面那一数，why？
                    %             反正就是将卷积核每个元素初始化为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
                    %             之间的随机数。因为这里是权值共享的，也就是对于一张特征map，所有感受野位置的卷积核都是一样的
                    %             所以只需要保存的是 inputmaps * outputmaps 个卷积核。
                    %          2）为什么这里是inputmaps * outputmaps个卷积核？
                    %%=========================================================================

                    %卷积核的初始化生成一个5*5的卷积核，值为-1~1之间的随机数
                    %再乘以sqrt(6/(7*25))，sqrt(6/(18*25))

                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end

                %偏置初始化为0，每个输出map只有一个bias，并非每个filter一个bias
                net.layers{l}.b{j} = 0;

            end
             
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    
    %%=========================================================================
    % 主要功能：初始化最后一层，也就是输出层的参数值
    % 算法流程：1）fvnum 是输出层的前面一层的神经元个数。这一层的上一层是经过pooling后的层，包含有inputmaps个
    %             特征map。每个特征map的大小是mapsize，所以，该层的神经元个数是 inputmaps * （每个特征map的大小）  
    %          2）onum 是标签的个数，也就是输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元
    %          3）net.ffb和net.ffW为最后一层（全连接层）的偏置和权重
    %%=========================================================================
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
