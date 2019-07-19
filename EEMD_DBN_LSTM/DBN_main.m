function pre_data = DBN_main(h,lag,left,right,s)

%tic
% lag = 3;
% d = 31;
train_input = [];
train_output = [];

for i = left:right-lag+1
    train_input = [train_input; i:i+lag-1]; 
    train_output = [train_output; s(i+lag+h-1)];
end

[train_input,min_input,max_input,train_output,min_output,max_output] = premnmx(train_input',train_output');
train_input = train_input';
train_output = train_output';

%% network setup
% 输入的是10维数据，所以为10，后面的DBN层数可以自己设置
dbn.sizes = [35 25];
opts.numepochs =  100;
opts.batchsize =  size(train_input,1);
opts.momentum  =   0;
opts.alpha     =   0.01;
dbn = dbnsetup(dbn, train_input, opts);
dbn = dbntrain(dbn, train_input, opts);

%% unfold dbn to nn
% 将DBN网络转化为NN网络，这里参数为1是因为输出维数为1
nn = dbnunfoldtonn(dbn, 1);
 
nn.activation_function = 'tanh_opt';    %  tanh_opt activation function
nn.output              = 'linear';      %  linear is usual choice for regression problems
nn.learningRate        = 0.001;         %  Linear output can be sensitive to learning rate
 
opts.numepochs = 100;   %  Number of full sweeps through data
opts.batchsize = size(train_input,1);   %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_input, train_output, opts);

%t2 = toc

% nnoutput calculates the predicted regression values
% test_input = [];
% for i = size(train_data,2)+h:h:size(train_data,2)+size(test_data,2) 
%     test_input = [test_input; i-h-lag+1:i-h]; 
% end
% 
% test_input = tramnmx(test_input',min_input,max_input);
% test_input = test_input';

test_input = i+1:i+lag; 
test_input = tramnmx(test_input',min_input,max_input);
test_input = test_input';

%tic

pre_data = nnoutput(nn, test_input);%预测结果

%t3=toc
pre_data = postmnmx(pre_data,min_output,max_output);

end

