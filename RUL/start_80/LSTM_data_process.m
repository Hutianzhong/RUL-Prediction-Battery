function [input,output] = LSTM_data_process(d,train_data_initial,lag)

input = [];
output = [];

for i = d:size(train_data_initial,2)-lag
    input = [input; train_data_initial(i:i+lag-1)]; 
    output = [output; train_data_initial(i+lag)];
end

end