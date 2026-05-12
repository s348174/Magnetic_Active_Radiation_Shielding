%% INTERAZIONE TOTALE: TRAIETTORIA CURVA + DISTORSIONE CAMPO (10 SUBPLOTS)
clc; clear;

% 1. Setup Griglia
x_vec = linspace(-50, 50, 80);
y_vec = linspace(-50, 50, 80);
[X, Y] = meshgrid(x_vec, y_vec);

% 2. Parametri Fisici Calibrati
m = [0, 5];          % Momento dipolo al centro (0,0)
q_eff = 0.8e+06;        % Carica "gonfiata" per vedere la perturbazione del campo
mass_eff = 1;          % Massa fittizia per rendere visibile la curva
mu0 = 4*pi*1e-7;
dt = 0.15;             % Passo temporale

% Stato iniziale protone
pos = [49, 4];          % Parte da destra, alto
vel = [-5.5, -0.2];    % Velocità iniziale verso sinistra
traj = pos;

% 3. Funzioni di calcolo Campo B
% Funzione Dipolo
calcB_dip = @(x, y) deal(...
    3*x.*y.*m(2) ./ (x.^2 + y.^2 + 1.2).^2.5, ...
    (3*y.^2.*m(2) ./ (x.^2 + y.^2 + 1.2).^2.5 - m(2) ./ (x.^2 + y.^2 + 1.2).^1.5));

% Funzione Campo Protone (Biot-Savart)
calcB_prot = @(px, py, vx, vy, X, Y) deal(...
    -mu0 * q_eff * vx * (Y-py) ./ (4*pi*((X-px).^2 + (Y-py).^2 + 0.8).^1.5), ...
     mu0 * q_eff * vx * (X-px) ./ (4*pi*((X-px).^2 + (Y-py).^2 + 0.8).^1.5));

figure('Color', 'k', 'Position', [50, 50, 1600, 700]);

for i = 1:10
    % --- FISICA DI LORENTZ (Calcolo traiettoria curva) ---
    % Eseguiamo piccoli step per una curva fluida
    for step = 1:8
        [bx_at_p, by_at_p] = calcB_dip(pos(1), pos(2));
        B_local_mag = sqrt(bx_at_p^2 + by_at_p^2);
        
        % Accelerazione di Lorentz (semplificata per piano 2D)
        % F = q * (v x B) -> In 2D genera una forza normale alla velocità
        accel = 0.15 * [vel(2), -vel(1)] * B_local_mag / mass_eff;
        
        vel = vel + accel * dt;
        pos = pos + vel * dt;
        traj = [traj; pos];
    end
    
    % --- CALCOLO CAMPI PER IL PLOT ---
    [Bx_dip, By_dip] = calcB_dip(X, Y);
    [Bx_prot, By_prot] = calcB_prot(pos(1), pos(2), vel(1), vel(2), X, Y);
    
    Bx_tot = Bx_dip + Bx_prot;
    By_tot = By_dip + By_prot;
    B_mag = sqrt(Bx_tot.^2 + By_tot.^2);
    
    % --- DISEGNO SUBPLOT ---
    subplot(2, 5, i); hold on;
    
    % Intensità totale (Mappa di calore)
    contourf(X, Y, B_mag, 35, 'LineColor', 'none');
    colormap(hot); shading interp;
    
    % Linee di forza (Vedi la deflessione e la distorsione simultanea)
    st = streamslice(X, Y, Bx_tot, By_tot, 1.4);
    set(st, 'Color', 'w', 'LineWidth', 0.8);
    
    % Protone e Traiettoria
    plot(traj(:,1), traj(:,2), 'c-', 'LineWidth', 1.5);
    plot(pos(1), pos(2), 'co', 'MarkerFaceColor', 'c', 'MarkerSize', 5);
    
    % Dipolo centrale
    plot(0, 0, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    
    % Estetica
    xlim([-10 10]); ylim([-10 10]);
    clim([0 0.5]); 
    title(['t = ', num2str(i)], 'Color', 'w');
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    axis equal tight;
end

sgtitle('Simulation: Magnetic field deformation by a proton', 'Color', 'w', 'FontSize', 15);