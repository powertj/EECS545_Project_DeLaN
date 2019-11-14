function l = stagecost(x, u, zg, Q, R)

e = x-zg;
l = e'*Q*e + u'*R*u;

end
