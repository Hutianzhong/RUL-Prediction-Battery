function RUL = threshold(x,ts)

for i = 1:size(x,2)
    if (x(i) < ts)
    RUL = i;
    break;
    end
end

RUL = i;

end

