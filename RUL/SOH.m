function [ts,s,c] = SOH(var)

var = var.B0005.cycle;
c = 0;
s = [];

for i = 1 : size(var,2)
    type = var(i).type;
    if type == "discharge"
        c = c + 1;
        cap = var(i).data.Capacity;
        s = [s;cap];
    end
end
    ts = 1.38/s(1);
    s = s/s(1);
end

