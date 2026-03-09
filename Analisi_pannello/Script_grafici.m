%% ANALISI SPOSTAMENTI PANNELLO TOROIDE
clc
clear
close all

data = readmatrix('Report_spostamenti_totali'); 
posizione = data(:,3); 
spostamento = data(:,5);

figure
plot(posizione, spostamento, 'k', 'LineWidth', 2);
grid on;
xlabel('cordinata y [mm]');
ylabel('Spostamento (mm)');
title('Spostamento sezione centrale');

scatter3(data(:,2),data(:,3),data(:,4),'.')
tri = delaunay(data(:,2),data(:,3));
trisurf(tri,data(:,2),data(:,3),data(:,4))
shading interp
colormap parula
view(3)

%% ANALISI STRESS PANNELLO TOROIDE
clc
clear
close all

data = readmatrix('Report_stress_bordo_freddo'); 
posizione = data(:, 3); 
stress_est = data(:, 5);
stress_int = data(:, 6);

figure
plot(posizione, stress_est, 'k', 'LineWidth', 1.25);
hold on
plot(posizione, stress_int, 'r', 'LineWidth', 1.25);
grid on;
legend('sup est','sup int');
xlabel('cordinata y [mm]');
ylabel('stress [MPa]');
title('Stress bordo (freddo)');

