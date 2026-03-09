function [Bx, By, Bz] = compute_B_field(R, I, r, rho)
    % Function to compute B vector components from coil of radius R and
    % current intensity I.
    % 
    % Input:
    %   R - Coil radius
    %   I - Current intensity
    %   r - Point of evaluation of B
    %   rho - Torus internal radius
    % 
    % Output:
    %   Bx, By, Bz - B vector components
    
    % Definition of mu constant
    mu0 = 4 * pi * 1e-7;
    x = r(1);
    y = r(2);
    z = r(3);

    % Check if point is inside torus
    if power(R - sqrt(x^2+y^2), 2)+z^2 < rho^2  
        Bx = 0;
        By = 0;
        Bz = 0;
        return;
    end

    N = 500; % Number of points
    theta = linspace(0, 2*pi, 2*N+1); % includes 0 and 2*pi (necessary due to symmetry)
    
    % Integration step
    dtheta = pi/N;
    
    % Integrand functions
    denom = (x.^2 + y.^2 + z.^2 + R^2 - 2*R*x.*cos(theta) - 2*R*y.*sin(theta)).^(3/2);
    fx = z .* cos(theta) ./ denom;
    fy = z .* sin(theta) ./ denom;
    fz = (R - x.*cos(theta) - y.*sin(theta)) ./ denom;
    
    % Composed Simpson's formula
    integral_x = (fx(1)+4*sum(fx(2:2:2*N))+ 2*sum(fx(3:2:2*N-1))+fx(2*N+1)) * dtheta/3;
    integral_y = (fy(1)+4*sum(fy(2:2:2*N))+ 2*sum(fy(3:2:2*N-1))+fy(2*N+1)) * dtheta/3;
    integral_z = (fz(1)+4*sum(fz(2:2:2*N))+ 2*sum(fz(3:2:2*N-1))+fz(2*N+1)) * dtheta/3;

    % Compute B field vector
    coeff = mu0 * I * R / (4 * pi);
    Bx = coeff * integral_x;
    By = coeff * integral_y;
    Bz = coeff * integral_z;
end