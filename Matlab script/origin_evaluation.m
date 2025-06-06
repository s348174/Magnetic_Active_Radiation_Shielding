%% Evaluate numbers of coils performances at origin
R = 15;
rho = 1;
I = 10000;
point = [0,0,0];
N_coils = (1000:100:10000)';
n_simulation = length(N_coils);
B_habitat = zeros(n_simulation, 1);
parfor i = 1:n_simulation
    torus = Torus(R, rho, I, N_coils(i));
    B_habitat(i) = norm(get_point_field(torus,point));
end

mu0 = 4 * pi * 1e-7;
real_value = mu0*I/(2*R);

abs_err = abs(B_habitat - real_value);
figure;
plot(N_coils,abs_err,'LineWidth',2,'Color','r');
title('Absolute error at origin relative to coils number ',  ['(R = ', int2str(R), ')']);
xlabel('Number of coils');
ylabel('Abs. err.');
grid on;

rel_err = abs_err/real_value;
figure;
plot(N_coils,rel_err,'LineWidth',2,'Color','k');
title('Relative error at origin relative to coils number ', ['(R = ', int2str(R), ')']);
xlabel('Number of coils');
ylabel('Rel. err.');
grid on;