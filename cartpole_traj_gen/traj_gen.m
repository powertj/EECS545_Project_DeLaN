add_casadi_mpctools;

mpc = import_mpctools();

% cartpole params (should match what's in run_mpc_sim
params.M = 10; % mass of cart
params.m = 1; % mass of pendulum
params.l = 1; % pendulum length
params.g = 9.81; % gravity

goal_x = 1:0.1:2;
Nsim = length(goal_x);
sim_length = 20*25;
trajectories = zeros(Nsim,sim_length,6);
H = zeros(Nsim,sim_length,2,2);
c = zeros(Nsim,sim_length,2,2);
g = zeros(Nsim,sim_length,2);
torques = zeros(Nsim,sim_length,2);
for k=1:length(goal_x)
    data = run_mpc_sim([goal_x(k); pi; 0; 0], sim_length, mpc);
    trajectories(k,:,1:2) = data.x(1:2,1:end-1).';
    for i=1:sim_length
        trajectories(k,i,3:6) = cartpole_dynamics(data.x(:,i), data.u(:,i), params).';
        torques(k,i,:) = [data.u(:,i) 0];
        g(k,i,2) = params.m * params.g * params.l * sin(trajectories(k,i,2));
        c(k,i,1,2) = -params.m * params.l * trajectories(k,i,4) * sin(trajectories(k,i,2));
        H(k,i,:,:) = [params.m + params.M, params.m * params.l * cos(trajectories(k,i,2)); ...
            params.m * params.l * cos(trajectories(k,i,2)), params.m * params.l^2];
    end
end

save('cartpole_traj_out.mat', 'trajectories', 'torques', 'g', 'H', 'c')

%% swingup

control_lim = 185:5:250;

Nsim = length(control_lim);
sim_length = 20*5;
trajectories = zeros(Nsim,sim_length,6);
H = zeros(Nsim,sim_length,2,2);
c = zeros(Nsim,sim_length,2,2);
g = zeros(Nsim,sim_length,2);
torques = zeros(Nsim,sim_length,2);
for k=1:Nsim
    data = run_mpc_sim([0; pi; 0; 0], sim_length, control_lim(k), mpc);
    trajectories(k,:,1:2) = data.x(1:2,1:end-1).';
    for i=1:sim_length
        trajectories(k,i,3:6) = cartpole_dynamics(data.x(:,i), data.u(:,i), params).';
        torques(k,i,:) = [data.u(:,i) 0];
        g(k,i,2) = params.m * params.g * params.l * sin(trajectories(k,i,2));
        c(k,i,1,2) = -params.m * params.l * trajectories(k,i,4) * sin(trajectories(k,i,2));
        H(k,i,:,:) = [params.m + params.M, params.m * params.l * cos(trajectories(k,i,2)); ...
            params.m * params.l * cos(trajectories(k,i,2)), params.m * params.l^2];
    end
end

save('cartpole_traj_out.mat', 'trajectories', 'torques', 'g', 'H', 'c')

%% full revolution

control_lim = 185:5:250;

Nsim = length(control_lim);
sim_length = 20*5;
trajectories = zeros(Nsim,sim_length,6);
H = zeros(Nsim,sim_length,2,2);
c = zeros(Nsim,sim_length,2,2);
g = zeros(Nsim,sim_length,2);
torques = zeros(Nsim,sim_length,2);
for k=1:Nsim
    data = run_mpc_sim([0; pi; 0; 0], sim_length, control_lim(k), mpc);
    trajectories(k,:,1:2) = data.x(1:2,1:end-1).';
    for i=1:sim_length
        trajectories(k,i,3:6) = cartpole_dynamics(data.x(:,i), data.u(:,i), params).';
        torques(k,i,:) = [data.u(:,i) 0];
        g(k,i,2) = params.m * params.g * params.l * sin(trajectories(k,i,2));
        c(k,i,1,2) = -params.m * params.l * trajectories(k,i,4) * sin(trajectories(k,i,2));
        H(k,i,:,:) = [params.m + params.M, params.m * params.l * cos(trajectories(k,i,2)); ...
            params.m * params.l * cos(trajectories(k,i,2)), params.m * params.l^2];
    end
end

save('cartpole_traj_out.mat', 'trajectories', 'torques', 'g', 'H', 'c')

