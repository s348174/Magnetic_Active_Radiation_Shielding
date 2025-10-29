function success_rate = particles_B_trajectory_simulation(R,rho,m,q,I,dt,N_max)
    % Function that simulates a swarm of N_max particles with random initial 
    % parameters and verifies how many of them hit the Torus.
    % Other inputs:
    % - m: mass of the particles swarm
    % - q: particles charge
    % - I: current intensity
    % - dt: integration step
    % - T_max: simulation max time

    % Sampling v0 modules from Maxwell-Boltzman distribution
    v_samples = sample_MB_speed(m,N_max);
    intersect_count = 0; % Hits counter 
    for i = 1:N_max
        % Sampling initial position x0 from uniform distribution on sphere
        % of radius 4R
        theta = unifrnd(0,2*pi);
        phi = unifrnd(0,pi);
        r0 = [4*R*sin(phi)*cos(theta),4*R*sin(phi)*sin(theta),4*R*cos(phi)];

        % Simulation a uniform random target inside torus
        theta = unifrnd(0,2*pi);
        traj_0 = [R*cos(theta),R*sin(theta),0]-r0;
        v0 = (v_samples(i)/norm(traj_0)).*traj_0; % Initial speed vector

        % Compute trajectory and verify if hits
        T_max = 8*R/v_samples(i);
        [r,shield] = compute_trajectory_B(r0,v0,q,m,R,rho,I,dt,T_max);
        if shield == false
            intersect_count = intersect_count + 1;      
        end
    end
    success_rate = 1-(intersect_count/N_max);
end