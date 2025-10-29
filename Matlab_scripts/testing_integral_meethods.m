% Parametri
R = 1;
x = 0.5; y = 0.3; z = 0.7;

% Integranda da testare
%f = @(theta) (z .* cos(theta)) ./ ((x.^2 + y.^2 + z.^2 + R^2 - 2*R*x.*cos(theta) - 2*R*y.*sin(theta)).^(3/2));
f = @(theta) sin(theta).^2;
f_true = @(theta) -sin(theta).*cos(theta)./2 + theta/2;
val_true = f_true(2*pi);

% ===========================
% Metodo 1: integral()
tic;
val_integral = integral(f, 0, 2*pi, 'AbsTol', 1e-12, 'RelTol', 1e-12);
t_integral = toc;

% ===========================
% Metodo 2: Simpson composta
N = 500;  % deve essere pari
theta = linspace(0, 2*pi, 2*N+1);
h = pi/N;

% Valutazione funzione
fx = f(theta);

% Coefficienti Simpson
coeff = ones(1, 2*N+1);
coeff(2:2:end-1) = 4;
coeff(3:2:end-2) = 2;

% Integrazione con Simpson
tic;
val_simpson = h/3 * sum(coeff .* fx);
t_simpson = toc;

% ===========================
% Output confronto
fprintf('Risultato integral()     = %.15f\n', val_integral);
fprintf('Risultato Simpson        = %.15f\n', val_simpson);
fprintf('Errore integral()        = %.2e\n', abs(val_integral - val_true));
fprintf('Errore simpson           = %.2e\n', abs(val_simpson - val_true));
fprintf('Tempo integral()         = %.4f s\n', t_integral);
fprintf('Tempo Simpson composto   = %.4f s\n', t_simpson);