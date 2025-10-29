%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFRONT VALUES OF B CALCULATED WITH DIFFERENT METHODS %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PARAMETERS INITIALIZATION
R = 10;
rho = 1.5;
I = 10000;
N = 1000;
x = linspace(0,2*R,N);

%% DISCRETE SIMULATION METHOD CALCULATIONS
coils = 360;
torus = Torus(R,rho,I,coils);
B_mag_disc = zeros(N,1);
parfor i = 1:N
    point = [x(i),0,0];
    B_mag_disc(i) = norm(get_point_field(torus, point));
end
%fprintf("Final B magnitudes array");
%disp(B_mag)

%% THEORICAL MODEL
B_mag_theo = zeros(N,1);
for i = 1:N
    point = [x(i), 0, 0];
    [Bx, By, Bz] = compute_B_field(R, I, point, rho);
    B = [Bx, By, Bz];
    B_mag_theo(i) = norm(B);
end
figure;
plot(x,B_mag_disc,'LineWidth',2,'Color','b');
hold on;
plot(x,B_mag_theo,'LineWidth',2,'Color','g');
legend({"Discrete simulation", "Theorical model"}, 'Location', 'Best');
title('Distribution of B magnitude along x-axis');
xlabel('X-axis');
ylabel('B magnitude');
grid on;


%% ABSOLUTE ERROR
err = zeros(N,1);
for i = 1:N
    err(i) = abs(B_mag_theo(i) - B_mag_disc(i));
end
figure;
plot(x, err, 'LineWidth', 2, 'Color', 'r');
title("Absolute error between simulations");
xlabel('X-axis');
ylabel('Abs. err.');
grid on;
hold off;

%% RELATIVE ERROR
rel_err = abs(B_mag_disc - B_mag_theo)./B_mag_disc;
figure;
plot(x, rel_err, 'LineWidth', 2, 'Color', 'k');
title("Relative error between simulations");
xlabel('X-axis');
ylabel('Rel. err.');
grid on;