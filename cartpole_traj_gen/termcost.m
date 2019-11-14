function Vf = termcost(x, zg, Pinf)
e = x-zg;
Vf = e'*Pinf*e;
end
