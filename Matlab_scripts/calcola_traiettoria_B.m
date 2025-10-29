function [r,shield] = calcola_traiettoria_B(r0, v0, q, m, R, rho, I, dt, T_max)
  % Calcola la traiettoria di una particella carica in un campo magnetico
  % generato da una spira di raggio R percorsa da corrente I.
  %
  % INPUT:
  % v0 - Velocità iniziale [vx0, vy0, vz0]
  % r0 - Posizione iniziale [x0, y0, z0]
  % q - Carica della particella
  % m - Massa della particella
  % R - Raggio della spira
  % rho - Spessore del toro
  % I - Corrente nella spira
  % dt - Passo temporale per l'integrazione
  % T_max - Tempo totale della simulazione
  %
  % OUTPUT:
  % r - Array Nx3 che contiene i punti della traiettoria della particella
  % shield - Booleano che indica, se false, che la traiettoria attraversa
  % il toro di raggio R e spessore rho

  % Shield è settato inizialmente come true
  shield = true;
  
  % Numero di passi temporali
  N = round(T_max / dt);

  % Inizializzazione dei vettori di posizione e velocità
  r = zeros(N, 3); % Posizione [x, y, z]
  v = zeros(N, 3); % Velocità [vx, vy, vz]

  % Condizioni iniziali
  r(1, :) = r0;
  v(1, :) = v0;
  
  % Loop temporale per integrare l'equazione del moto
  for i = 1:N-1
    % Calcola il campo magnetico nel punto corrente
    [Bx, By, Bz] = calcola_campo_B(R, I, r(i,:),rho);
    B = [Bx, By, Bz];

    % Calcola la forza di Lorentz: F = q * (v x B)
    F = q.*cross(v(i, :), B);

    % Calcola l'accelerazione: a = F / m
    a = F./m;

    % Aggiorna velocità e posizione
    v(i+1, :) = v(i, :) + dt.*a;
    r(i+1, :) = r(i, :) + dt.*(v(i+1, :)+v(i, :))./2;
    
    % Se la traiettoria entra nel toro, shield viene settato a false
    x_t = r(i+1,1);
    y_t = r(i+1,2);
    z_t = r(i+1,3);
    if power(R - sqrt(x_t^2+y_t^2), 2)+z_t^2 < rho
        shield = false;
    end

    %display(B, 'field');
    %display(F, 'force');
    %display(a,'acc.');
    %display(v(i+1,:), 'speed');
    %display(r(i+1,:), 'pos.');
  end
end