add_casadi_mpctools;

mpc = import_mpctools();

goal_x = 1:0.1:2;
sim_length = 20*25;
trajs = zeros(Nsim,sim_length,6);
Hs = zeros(Nsim,sim_length,2,2);
Gs = zeros(Nsim,sim_length,2);
Us = zeros(Nsim,sim_length,2);
for k=1:length(goal_x)
    data = run_mpc_sim([goal_x(k); pi; 0; 0], sim_length, mpc);
    trajs(k,:,1:2) = data.x(1:2,1:end-1).';
    for i=1:sim_length
        trajs(k,i,3:6) = cartpole_dynamics(data.x(:,i), data.u(:,i), params).';
        Us(k,i,:) = [data.u(:,i) 0];
        Gs(k,i,2) = params.m * params.g * params.l * sin(trajs(k,i,2));
        Hs(k,i,:,:) = [params.m + params.M, params.m * params.l * cos(trajs(k,i,2)); ...
            params.m * params.l * cos(trajs(k,i,2)), params.m * params.l^2];
    end
end

save('cartpole_traj_out.mat', 'trajs', 'Us', 'Gs', 'Hs')
