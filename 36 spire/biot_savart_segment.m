function B = biot_savart_segment(P, A, B, I)
    % P: Punto dove calcolare il campo [x, y, z]
    % A, B: Estremi del segmento [x, y, z]
    % I: Corrente in Ampere
    
    mu0 = 4*pi*1e-7;
    r1 = P - A;
    r2 = P - B;
    dL = B - A;
    
    % Cross product e distanze
    cross_r1r2 = cross(r1, r2);
    dist_r1 = norm(r1);
    dist_r2 = norm(r2);
    
    % Formula di Biot-Savart per un segmento finito
    if norm(cross_r1r2) < 1e-12 % Evita divisione per zero se P è sul filo
        B = [0, 0, 0];
    else
        term1 = (mu0 * I) / (4 * pi * norm(cross_r1r2)^2);
        term2 = dot(dL, (r1/dist_r1 - r2/dist_r2));
        B = term1 * term2 * cross_r1r2;
    end
end