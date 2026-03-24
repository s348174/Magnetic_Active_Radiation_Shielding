%% 36 SPIRE
clc
clear
close all

Raggio_toro_g = 8.5; % Raggio grande del toroide
RAGGIO_TORO = 1.5; % Raggio della sezione del toroide
raggi   = [10, 7, 8.5, 8.5, Raggio_toro_g + (RAGGIO_TORO*cos(2*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(2*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-2*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-2*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(4*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(4*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-4*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-4*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(6*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(6*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-6*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-6*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(8*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(8*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-8*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-8*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(10*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(10*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-10*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-10*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(12*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(12*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-12*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-12*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(14*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(14*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-14*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-14*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(16*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(16*pi/36)), Raggio_toro_g + (RAGGIO_TORO*cos(-16*pi/36)), Raggio_toro_g - (RAGGIO_TORO*cos(-16*pi/36))];    % Raggi delle 36 spire (metri)
quote   = [0, 0, 1.5, -1.5, RAGGIO_TORO*sin(2*pi/36), -RAGGIO_TORO*sin(2*pi/36), RAGGIO_TORO*sin(-2*pi/36), -RAGGIO_TORO*sin(-2*pi/36), RAGGIO_TORO*sin(4*pi/36), -RAGGIO_TORO*sin(4*pi/36), RAGGIO_TORO*sin(-4*pi/36), -RAGGIO_TORO*sin(-4*pi/36), RAGGIO_TORO*sin(6*pi/36), -RAGGIO_TORO*sin(6*pi/36), RAGGIO_TORO*sin(-6*pi/36), -RAGGIO_TORO*sin(-6*pi/36), RAGGIO_TORO*sin(8*pi/36), -RAGGIO_TORO*sin(8*pi/36), RAGGIO_TORO*sin(-8*pi/36), -RAGGIO_TORO*sin(-8*pi/36), RAGGIO_TORO*sin(10*pi/36), -RAGGIO_TORO*sin(10*pi/36), RAGGIO_TORO*sin(-10*pi/36), -RAGGIO_TORO*sin(-10*pi/36), RAGGIO_TORO*sin(12*pi/36), -RAGGIO_TORO*sin(12*pi/36), RAGGIO_TORO*sin(-12*pi/36), -RAGGIO_TORO*sin(-12*pi/36), RAGGIO_TORO*sin(14*pi/36), -RAGGIO_TORO*sin(14*pi/36), RAGGIO_TORO*sin(-14*pi/36), -RAGGIO_TORO*sin(-14*pi/36), RAGGIO_TORO*sin(16*pi/36), -RAGGIO_TORO*sin(16*pi/36), RAGGIO_TORO*sin(-16*pi/36), -RAGGIO_TORO*sin(-16*pi/36)];     % Altezze Z delle 36 spire (metri)
correnti = [100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36, 100000/36]; % Corrente per ogni spira (Ampere)

N_seg = 100; % Numero di segmenti per spira 

% Definizione griglia di calcolo
[X, Y, Z] = meshgrid(-20:1:20, -20:1:20, -10:1:15); 
Bx = zeros(size(X)); 
By = zeros(size(X)); 
Bz = zeros(size(X));

%Calcolo del campo 
for i = 1:numel(X)
    P = [X(i), Y(i), Z(i)];
    B_total = [0, 0, 0];
    
    % Consideriamo i contributi delle 36 spire
    for s = 1:length(raggi)
        R_curr = raggi(s);
        Z_curr = quote(s);
        I_curr = correnti(s);
        
        % Generazione nodi per la spira considerata
        theta = linspace(0, 2*pi, N_seg + 1);
        
        nodes = [R_curr*cos(theta)', R_curr*sin(theta)', Z_curr*ones(size(theta'))];
        
        % Calcolo Biot-Savart per la spira s-esima
        for j = 1:N_seg
            A = nodes(j, :);
            B = nodes(j+1, :);
            B_total = B_total + biot_savart_segment(P, A, B, I_curr);
        end
    end
    
    Bx(i) = B_total(1); 
    By(i) = B_total(2); 
    Bz(i) = B_total(3);
end

figure(1);
[sx, sy, sz] = meshgrid(-12:4:12, -12:4:12, quote); 
h = streamline(X, Y, Z, Bx, By, Bz, sx, sy, sz);
set(h, 'Color', 'red', 'LineWidth', 1);
grid on;
axis equal;
title('Campo Magnetico (36 Spire)');
hold on
% Disegniamo graficamente le 36 spire per riferimento
t_plot = linspace(0, 2*pi, 100);
for s = 1:length(raggi)
    plot3(raggi(s)*cos(t_plot), raggi(s)*sin(t_plot), quote(s)*ones(size(t_plot)), 'k-', 'LineWidth', 2);
end

view(3);
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');

%Calcola il modulo del campo
B_mag = sqrt(Bx.^2 + By.^2 + Bz.^2);

%Trovare l'indice centrale della PRIMA dimensione (Y)
fetta = round(size(Y, 1) / 2); 

%Estraiamo le sezioni correggendo le dimensioni con squeeze
X_sezione = squeeze(X(fetta, :, :)); 
Z_sezione = squeeze(Z(fetta, :, :));
B_sezione = squeeze(B_mag(fetta, :, :)); 

figure(2)
contourf(X_sezione, Z_sezione, B_sezione, 20, 'LineColor', 'none')
colorbar;
colormap(jet);
axis equal;
grid on;
title(['Sezione del Campo Magnetico al centro (Y = ', num2str(Y(fetta,1,1)), ' m)']);
xlabel('X (metri)');
ylabel('Z (metri)');