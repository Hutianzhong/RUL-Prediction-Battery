clear all;
close all;
clc;

str = '/Users/hutianzhong/Desktop/EMD_DBN_LSTM/数据/B0005.mat';
var = load(str);
[ts,s,cycle] = SOH(var);

%num_train = 80;
h = 1;

ans_start_100 = load('/Users/hutianzhong/Desktop/EMD_DBN_LSTM/#5/RUL/start_100/result/ans.mat');
ans_start_100 = sum(ans_start_100.ans);

ans_start_90 = load('/Users/hutianzhong/Desktop/EMD_DBN_LSTM/#5/RUL/start_90/result/ans.mat');
ans_start_90 = sum(ans_start_90.ans);

ans_start_80 = load('/Users/hutianzhong/Desktop/EMD_DBN_LSTM/#5/RUL/start_80/result/ans.mat');
ans_start_80 = sum(ans_start_80.ans);

%ts = 1.38/2;

start_100_real_RUL = threshold(s(100+h:h:size(s,1))',ts);
start_90_real_RUL = threshold(s(90+h:h:size(s,1))',ts);
start_80_real_RUL = threshold(s(80+h:h:size(s,1))',ts);
start_100_pre_RUL = threshold(ans_start_100,ts);
start_90_pre_RUL = threshold(ans_start_90,ts);
start_80_pre_RUL = threshold(ans_start_80,ts);

gap_100 = abs(start_100_real_RUL - start_100_pre_RUL)
gap_90 = abs(start_90_real_RUL - start_90_pre_RUL)
gap_80 = abs(start_80_real_RUL - start_80_pre_RUL)

figure
title('总预测')
hold on
plot(1:size(s,1),s, 'g-', 'linewidth', 2);
plot(100+h:100+start_100_pre_RUL,ans_start_100(1:start_100_pre_RUL), 'r-.', 'linewidth', 2);
plot(90+h:90+start_90_pre_RUL,ans_start_90(1:start_90_pre_RUL), 'b-.', 'linewidth', 2);
plot(80+h:80+start_80_pre_RUL,ans_start_80(1:start_80_pre_RUL), 'm-.', 'linewidth', 2);
plot([1 size(s,1)],[ts ts],'k-','LineWidth',2);

plot(80,ans_start_80(1),'s','color','m');
plot(80+start_80_pre_RUL,ans_start_80(start_80_pre_RUL),'s','color','m');
plot(90,ans_start_90(1),'s','color','b');
plot(90+start_90_pre_RUL,ans_start_90(start_90_pre_RUL),'s','color','b');
plot(100,ans_start_100(1),'s','color','r');
plot(100+start_100_pre_RUL,ans_start_100(start_100_pre_RUL),'s','color','r');

plot(100:100+start_100_pre_RUL,ones(size(100:100+start_100_pre_RUL,2))*ans_start_100(1),'color','r');
t=size([ans_start_100(end):0.01:ans_start_100(1)+0.05],2);
plot((100+start_100_pre_RUL)*ones(t,1),ans_start_100(end):0.01:ans_start_100(1)+0.05,'color','r');
text(100+start_100_pre_RUL,ans_start_100(1)+0.01,'End of Prediction');

plot(90:90+start_90_pre_RUL,ones(size(90:90+start_90_pre_RUL,2))*ans_start_90(1),'color','b');
t=size([ans_start_90(end):0.01:ans_start_90(1)+0.05],2);
plot((90+start_90_pre_RUL)*ones(t,1),ans_start_90(end):0.01:ans_start_90(1)+0.05,'color','b');
text(90+start_90_pre_RUL,ans_start_90(1)+0.01,'End of Prediction');

plot(80:80+start_80_pre_RUL,ones(size(80:80+start_80_pre_RUL,2))*ans_start_80(1),'color','m');
t=size([ans_start_80(end):0.01:ans_start_80(1)+0.05],2);
plot((80+start_80_pre_RUL)*ones(t,1),ans_start_80(end):0.01:ans_start_80(1)+0.05,'color','m');
text(80+start_80_pre_RUL,ans_start_80(1)+0.01,'End of Prediction');

% plot(num_train+h:h:size(s,1),ans_start_70, 'd-','color','m', 'linewidth', 1);
% plot(num_train+h:h:size(s,1),ans_start_60, '^-','color','c', 'linewidth', 1);
plot([100 100],[0.7 0.9],'r-','LineWidth',2);
plot([90 90],[0.7 0.9],'b-','LineWidth',2);
plot([80 80],[0.7 0.9],'m-','LineWidth',2);

xlabel('Time(cycle)');
ylabel('SOH');

legend({ '真实值', 'start(100)','start(90)','start(80)','失效阈值'});

