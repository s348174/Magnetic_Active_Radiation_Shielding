%% This script launches a Montecarlo simulation of a particles swarm directed 
%% towards the torus

addpath '../trajectory_computation'

R = 20;
rho = 1.5;
m = 1.67e-27;
q = 1.6e-19;
I = 1e4;
dt = 1e-3;
N_max = 1e3;
success_rate = particles_B_trajectory_simulation(R,rho,m,q,I,dt,N_max);
display(success_rate);