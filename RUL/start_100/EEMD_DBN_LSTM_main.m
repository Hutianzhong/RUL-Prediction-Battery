clear all;
close all;
clc;

str = '/Users/hutianzhong/Desktop/EMD_DBN_LSTM/数据/B0005.mat';
var = load(str);
[s,cycle] = SOH(var);

Nstd = 0.1;
NE = 100;

before_imf = eemd(s,Nstd,NE)';

dim = size(before_imf,1);

figure
title('eemd结果')
for i = 1:dim
subplot(dim,1,i);plot(before_imf(i,:));
end

c = corrcoef(before_imf');

after_imf = [before_imf(end,:)];

for i = dim-1:-1:2
    if abs(c(i,1)) > 0.2
        after_imf(end,:) = after_imf(end,:) + before_imf(i,:);
    else
        after_imf = [before_imf(i,:); after_imf];
    end
end

after_imf = [before_imf(1,:); after_imf];

dim = size(after_imf,1);

figure
title('相关性分析之后结果')
for i = 1:dim
subplot(dim,1,i);plot(after_imf(i,:));
end

ans = [];
h = 1;
num_train = 100;

ans = [ans; DBN_main(h,after_imf(end,1:num_train),after_imf(end,num_train+1:end))'];
figure
title('DBN预测')
hold on
plot(1:size(s,1),after_imf(end,:), 'o-', 'color','r', 'linewidth', 1);
plot(num_train+h:h:size(s,1),ans,'*-','color','b','linewidth', 1);
plot([num_train num_train],[0.6 0.9],'g-','LineWidth',4);
legend({ '真实值', '预测值'});

d = [51 31 31 31]; %

for i = 2:dim-1
    ans = [ans; LSTM_main(d(i-1),h,after_imf(i,1:num_train),after_imf(i,num_train+1:end))];
end

pre = sum(ans);

%% 画图
figure
title('总预测')
hold on
plot(1:size(s,1),s, 'o-', 'color','r', 'linewidth', 1);
plot(num_train+h:h:size(s,1),pre, '*-','color','b', 'linewidth', 1);
plot([num_train num_train],[0.6 0.9],'g-','LineWidth',4);
legend({ '真实值', '预测值'});

rmse = RMSE(pre,s(num_train+h:h:size(s,1))')
mape = MAPE(pre,s(num_train+h:h:size(s,1))')
mae = MAE(pre,s(num_train+h:h:size(s,1))')

save('c.mat','c');
save('rmse.mat','rmse');
save('mape.mat','mape');
save('mae.mat','mae');
save('ans.mat','ans');