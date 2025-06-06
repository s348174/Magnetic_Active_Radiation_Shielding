%% Calculate B field inside habitat with respect to increasing torus radius
rho = 1;
I = 10000;
N_coils = 360;
R = linspace(10, 1000, 1000);
n_simulation = length(R);
B_habitat = zeros(n_simulation, 1);
parfor i = 1:n_simulation
    point = [R(i), 0, 0];
    torus = Torus(R(i), rho, I, N_coils);
    B_habitat(i) = norm(get_point_field(torus,point));
end
figure;
plot(R,B_habitat,'LineWidth',2,'Color','b');
title('Simulation of B inside habitat relative to radius');
xlabel('Radius');
ylabel('B inside habitat');
grid on;