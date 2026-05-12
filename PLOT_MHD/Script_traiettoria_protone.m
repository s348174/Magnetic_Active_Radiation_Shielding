%% TRAIETTORIA DI UN PROTONE VERSO UN DIPOLO (LORENTZ FORCE)
clc; clear;

% 1. Setup Griglia per il Background (Campo B)
[X, Y] = meshgrid(linspace(-10, 10, 50), linspace(-10, 10, 50));
m = [0, 5]; % Momento magnetico del dipolo al centro (0,0)

% Funzione per calcolare il campo B in ogni punto
calcB = @(x, y) deal(...
    3*x.*y.*m(2) ./ (x.^2 + y.^2 + 0.5).^2.5, ...
    (3*y.^2.*m(2) ./ (x.^2 + y.^2 + 0.5).^2.5 - m(2) ./ (x.^2 + y.^2 + 0.5).^1.5));

[Bx_bg, By_bg] = calcB(X, Y);
B_mag = sqrt(Bx_bg.^2 + By_bg.^2);

% 2. Parametri del Protone (H+)
q = 1.6e-19;          % Carica (reale, ma scalata per simulazione)
mass = 1.6e-27;       % Massa (reale, ma scalata)
dt = 0.05;            % Passo temporale
pos = [8, 2];         % Posizione iniziale (arriva da destra)
vel = [-1.5, -0.2];   % Velocità iniziale (verso il dipolo)
traj = pos;           % Storico della traiettoria

% 3. Loop di Simulazione e Plot (10 Subplot)
figure('Color', 'k', 'Position', [50, 50, 1500, 600]);

for i = 1:10
    % Calcolo della traiettoria per 15 step tra ogni frame
    for step = 1:15
        % 1. Leggi il campo B nella posizione attuale della particella
        [bx_p, by_p] = calcB(pos(1), pos(2));
        
        % 2. Forza di Lorentz: F = q * (v x B) 
        % In 2D (piano xy), v x B ha solo componente Z, ma qui simuliamo la deflessione nel piano
        % Usiamo una versione semplificata della forza centripeta magnetica:
        accel = (q/mass) * [vel(2)*0.1, -vel(1)*0.1] * sqrt(bx_p^2 + by_p^2) * 1e-8; % Scaling per viz
        
        % 3. Aggiorna Velocità e Posizione
        vel = vel + accel * dt;
        pos = pos + vel * dt;
        traj = [traj; pos]; % Salva punto per il disegno
    end
    
    % --- DISEGNO SUBPLOT ---
    subplot(2, 5, i);
    hold on;
    
    % Campo B di sfondo
    contourf(X, Y, B_mag, 20, 'LineColor', 'none'); 
    colormap(hot); shading interp;
    
    % Linee di campo statiche
    streamslice(X, Y, Bx_bg, By_bg, 1.5);
    set(gca, 'Children', circshift(get(gca, 'Children'), -1)); % Manda linee in secondo piano
    
    % Traiettoria del Protone
    plot(traj(:,1), traj(:,2), 'c', 'LineWidth', 2); % Scia azzurra
    plot(pos(1), pos(2), 'co', 'MarkerFaceColor', 'c', 'MarkerSize', 5); % Testa del protone
    
    % Dipolo al centro
    plot(0, 0, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    
    % Estetica
    xlim([-10 10]); ylim([-10 10]);
    title(['t = ', num2str(i)], 'Color', 'w');
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    axis equal tight;
end

sgtitle('Deflessione di un Protone nel campo di un Dipolo (Forza di Lorentz)', 'Color', 'w', 'FontSize', 16);