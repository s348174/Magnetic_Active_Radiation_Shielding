R = 15;
rho = 1;
I = 10000;
point = [R,0,0];
N_coils = (100:1000:100000)';
n_simulation = length(N_coils);
B_habitat = zeros(n_simulation, 1);
for i = 1:n_simulation
    disp(i);
    torus = Torus(R, rho, I, N_coils(i));
    B_habitat(i) = norm(get_point_field(torus,point));
end

figure;
plot(N_coils,B_habitat,'LineWidth',2,'Color','b');
title('Simulation of B inside habitat relative to coils number');
xlabel('Number of coils');
ylabel('B inside habitat');
grid on;