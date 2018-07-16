%测试当前模型的准确率
function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, h] = max(net.o);   %输出行中最大概率值的位置
    [~, a] = max(y);           %标签中1的位置
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
    % numel(bad) return num of bad   
    % size(y,2)返回y的列数 
end
