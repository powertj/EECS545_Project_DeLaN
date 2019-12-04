function zdot = cartpole_dynamics(z,u,params)
%CARTPOLE_DYNAMICS given state z and control u computes zdot
% z = [x theat xdot thetadot].'
%
% params requires fields M, m, l, g corresponding to cart mass, pendulum
% mass, pendulum length, and gravity
%
% theta = 0 is hanging straight down
%
% copied from https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf

M = params.M;
m = params.m;
l = params.l;
g = params.g;

x_ddot = (u + m * sin(z(2)) * (l*z(4)^2 + g*cos(z(2)))) ...
    / (M + m*sin(z(2))^2);
theta_ddot = (-u*cos(z(2)) - m*l*z(4)^2*cos(z(2))*sin(z(2)) - (M + m)*g*sin(z(2))) ...
    / (l * (M + m*sin(z(2))^2));

zdot = [z(3); z(4); x_ddot; theta_ddot];

end

