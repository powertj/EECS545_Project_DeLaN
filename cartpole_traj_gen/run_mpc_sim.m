function data = run_mpc_sim(goal,sim_length,c,mpc)
%RUN_MPC_SIM 

% cartpole params
params.M = 10; % mass of cart
params.m = 1; % mass of pendulum
params.l = 1; % pendulum length
params.g = 9.81; % gravity

% initial state
z0 = [0 2*pi 0 0].';

% final state
zg = goal;

% controller frequency
Ts = 1/20;

% model
Nx = 4;
Nu = 1;
Nt = 30; % prediction horizon

% build system model
dint = mpc.getCasadiIntegrator(@(x,u) cartpole_dynamics(x,u,params), Ts, [Nx,Nu], {'x', 'u'}, {'dint'});
kwargs = struct('funcname', 'ode', 'rk4', true(), 'Delta', Ts);
Nc = 0;

% controller model (nonlinear and linear)
fnonlin = mpc.getCasadiFunc(@(x,u) cartpole_dynamics(x,u,params), [Nx, Nu], {'x', 'u'}, '**', kwargs);
linmodel = mpc.getLinearizedModel(fnonlin, {zg, zeros(Nu, 1)}, {'A', 'B'});

% define costs
Q = diag([10, 150, 0.01, 0.01]);
R = 0.005;
[Pinf, ~, ~] = dare(linmodel.A,linmodel.B,Q,R);
l = mpc.getCasadiFunc(@(x,u) stagecost(x,u,zg,Q,R), [Nx, Nu], {'x', 'u'}, {'l'});
Vf = mpc.getCasadiFunc(@(x) termcost(x,zg,Pinf), [Nx], {'x'}, {'Vf'});

% define constraints
lb = struct();
lb.x = [-Inf*ones(1,Nt+1);pi/2*ones(1,Nt+1);-20*ones(2,Nt+1)];
lb.u = -c*ones(Nu,Nt);
ub = struct();
ub.x = [Inf*ones(1,Nt+1);3*pi*ones(1,Nt+1);20*ones(2,Nt+1)];
ub.u = c*ones(Nu,Nt);

% build solvers
N = struct('x', Nx, 'u', Nu, 't', Nt);
kwargs = struct('l', l, 'Vf', Vf, 'lb', lb, 'ub', ub, 'verbosity', 0);

NLMPC_solver = mpc.nmpc('f', fnonlin, 'N', N, 'Delta', Ts, '**', kwargs);

% simulate closed loop
x = NaN(Nx, sim_length+1);
x(:,1) = z0;
u = NaN(Nu, sim_length);
for k=1:sim_length
    NLMPC_solver.fixvar('x', 1, x(:,k));
    NLMPC_solver.solve();
    if ~isequal(NLMPC_solver.status, 'Solve_Succeeded')
        warning('Solver failed at time %d!', k);
        break
    end
    u(:,k) = NLMPC_solver.var.u(:,1);
    x(:,k+1) = full(dint(x(:,k), u(:,k))); % simulate underlying dynamics
end
data = struct('x', x, 'u', u, 't', Ts*(0:sim_length));

if 1  % change to 0 to not plot
    figure(1)
    clf
    title('Nonlinear MPC cartpole')

    subplot(2,1,1)
    hold on
    %plot([0 100 100 200], [0 0 0.5 0.5; 0 0 0 0; 0 0 -0.5 -0.5], 'k--', 'LineWidth', 1.5)
    p = plot(data.t, data.x, 'LineWidth', 2);
    legend(p, {'$$x$$', '$$\theta$$', '$$\dot{x}$$', '$$\dot{\theta}$$'}, 'Interpreter', 'latex', 'FontSize', 12)
    xlabel('time (s)')

    subplot(2,1,2)
    hold on
    %plot([0 200 NaN 0 200], [0.03 0.03 NaN -0.03 -0.03], 'k--', 'LineWidth', 1.5)
    p = plot(data.t(1:sim_length), data.u, 'LineWidth', 2);
    legend(p, {'u'}, 'Interpreter', 'latex', 'FontSize', 12)
    xlabel('time (s)')
end

end

