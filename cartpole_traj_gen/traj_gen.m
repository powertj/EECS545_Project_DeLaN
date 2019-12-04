add_casadi_mpctools;

mpc = import_mpctools();

% cartpole params (should match what's in run_mpc_sim
params.M = 10; % mass of cart
params.m = 1; % mass of pendulum
params.l = 1; % pendulum length
params.g = 9.81; % gravity


%%
goal_x = 1:0.1:2;
Nsim = length(goal_x);
sim_length = 20*5;
trajectories = zeros(Nsim,sim_length,6);
H = zeros(Nsim,sim_length,2,2);
c = zeros(Nsim,sim_length,2,2);
g = zeros(Nsim,sim_length,2);
torques = zeros(Nsim,sim_length,2);
for k=1:length(goal_x)
    data = run_mpc_sim([goal_x(k); pi; 0; 0], sim_length, 200, mpc);
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


%% moving backwards

goal_x = -2:0.1:-1;
Nsim = length(goal_x);
sim_length = 20*5;
trajectories = zeros(Nsim,sim_length,6);
H = zeros(Nsim,sim_length,2,2);
c = zeros(Nsim,sim_length,2,2);
g = zeros(Nsim,sim_length,2);
torques = zeros(Nsim,sim_length,2);
for k=1:length(goal_x)
    data = run_mpc_sim([goal_x(k); pi; 0; 0], sim_length, 200, mpc);
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


%% combine into one file - WIP (need to zero pad)

clear

load('cartpole_trajs_goal_1_to_2.mat')

translate_tjs = trajectories;
translate_tqs = torques;
translate_g = g;
translate_H = H;
translate_c = c;

load('cartpole_trajs_swingup.mat')

swingup_tjs = trajectories;
swingup_tqs = torques;
swingup_g = g;
swingup_H = H;
swingup_c = c;

load('cartpole_trajs_revolution.mat')

rev_tjs = trajectories;
rev_tqs = torques;
rev_g = g;
rev_H = H;
rev_c = c;

load('cartpole_trajs_goal_neg2_to_neg1.mat')

backwards_tjs = trajectories;
backwards_tqs = torques;
backwards_g = g;
backwards_H = H;
backwards_c = c;

labels = [ones(1,size(translate_tjs,1)) 2*ones(1,size(swingup_tjs,1)) 3*ones(1,size(rev_tjs,1)) 4*ones(1,size(backwards_tjs,1))];

trajectories = cat(1,translate_tjs,swingup_tjs,rev_tjs,backwards_tjs);
torques = cat(1,translate_tqs,swingup_tqs,rev_tqs,backwards_tqs);
g = cat(1,translate_g,swingup_g,rev_g,backwards_g);
H = cat(1,translate_H,swingup_H,rev_H,backwards_H);
c = cat(1,translate_c,swingup_c,rev_c,backwards_c);

save('cartpole_all.mat', 'trajectories', 'torques', 'g', 'H', 'c', 'labels')

