clear all;
close all;
clc;

str = 'B0005.mat';
var = load(str);
[s,cycle] = SOH(var.B0005.cycle);

figure
title('电池容量退化趋势')
hold on
plot(1:size(s,1),s, '-', 'color','m', 'linewidth', 2);

str = 'B0006.mat';
var = load(str);
[s,cycle] = SOH(var.B0006.cycle);

hold on
plot(1:size(s,1),s, '-', 'color','b', 'linewidth', 2);

str = 'B0007.mat';
var = load(str);
[s,cycle] = SOH(var.B0007.cycle);

hold on
plot(1:size(s,1),s, '-', 'color','r', 'linewidth', 2);

str = 'B0018.mat';
var = load(str);
[s,cycle] = SOH(var.B0018.cycle);

hold on
plot(1:size(s,1),s, '-', 'color','g', 'linewidth', 2);

xlabel('Time/cycle');
ylabel('Capacity/Ah');

legend({ '#5 Battery','#6 Battery','#7 Battery','#18 Battery'});