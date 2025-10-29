function [Bx, By, Bz] = calcola_campo_B(R, I, r, rho)
    % Funzione per calcolare le componenti del campo magnetico Bx, By, Bz
    % generate da una spira di raggio R percorsa da corrente I.
    % 
    % Input:
    %   R - Raggio della spira
    %   I - Corrente nella spira
    %   r - Coordinate del punto in cui calcolare B
    %   rho - Spessore del toro
    % a
    % Output:
    %   Bx, By, Bz - Componenti del campo magnetico
    
    % Definizione della permeabilità magnetica del vuoto
    mu0 = 4 * pi * 1e-7;
    x = r(1);
    y = r(2);
    z = r(3);

    % Controllo se il punto è interno al toro
    if power(R - sqrt(x^2+y^2), 2)+z^2 < rho^2  
        Bx = 0;
        By = 0;
        Bz = 0;
        return;
    end

    % Denominatore comune
    % denom = @(theta) (x.^2 + y.^2 + z.^2 + R^2 - ...
    %                  2*R*x.*cos(theta) - 2*R*y.*sin(theta)).^(3/2);

    % Integrande
    %integrand_x = @(theta) z .* cos(theta) ./ denom(theta);
    %integrand_y = @(theta) z .* sin(theta) ./ denom(theta);
    %integrand_z = @(theta) (R - x.*cos(theta) - y.*sin(theta)) ./ denom(theta);

    % Integrazione numerica
    %integral_x = integral(integrand_x, 0, 2*pi, 'ArrayValued',true);
    %integral_y = integral(integrand_y, 0, 2*pi, 'ArrayValued',true);
    %integral_z = integral(integrand_z, 0, 2*pi, 'ArrayValued',true);

    N = 500; % Numero di punti
    theta = linspace(0, 2*pi, 2*N+1); % include 0 e 2*pi
    
    % Passo di integrazione
    dtheta = pi/N;
    
    % Funzioni integrande
    denom = (x.^2 + y.^2 + z.^2 + R^2 - 2*R*x.*cos(theta) - 2*R*y.*sin(theta)).^(3/2);
    fx = z .* cos(theta) ./ denom;
    fy = z .* sin(theta) ./ denom;
    fz = (R - x.*cos(theta) - y.*sin(theta)) ./ denom;
    
    % Formula di Simpson composta
    integral_x = (fx(1)+4*sum(fx(2:2:2*N))+ 2*sum(fx(3:2:2*N-1))+fx(2*N+1)) * dtheta/3;
    integral_y = (fy(1)+4*sum(fy(2:2:2*N))+ 2*sum(fy(3:2:2*N-1))+fy(2*N+1)) * dtheta/3;
    integral_z = (fz(1)+4*sum(fz(2:2:2*N))+ 2*sum(fz(3:2:2*N-1))+fz(2*N+1)) * dtheta/3;

    % Calcolo delle componenti del campo B
    coeff = mu0 * I * R / (4 * pi);
    Bx = coeff * integral_x;
    By = coeff * integral_y;
    Bz = coeff * integral_z;
end