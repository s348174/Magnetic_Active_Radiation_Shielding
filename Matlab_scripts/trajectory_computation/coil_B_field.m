function B_phi = coil_B_field(R, I)
    % Funzione per calcolare il campo B_phi in funzione di rho
    % Input:
    %   R - Raggio della spira
    %   I - Corrente nella spira
    % Output:
    %   B_phi - Valori del campo B_phi calcolati per diversi valori di rho

    % Definizione della permeabilità magnetica del vuoto
    mu0 = 4 * pi * 1e-7;
    phi = 0; % Assegnato a 0 come da richiesta

    % Definizione del range di rho
    rho_values = linspace(0, 3*R/2, 10000); % Variazione di rho tra 0 e 30
    B_values = zeros(size(rho_values)); % Array per memorizzare i risultati

    % Funzione da integrare
    integrand = @(theta, rho) (R - rho .* cos(theta - phi)) ./ ((R^2 + rho.^2 - 2*R*rho .* cos(theta - phi)).^(3/2));

    % Calcolo numerico dell'integrale per ogni valore di rho
    for i = 1:length(rho_values)
        rho = rho_values(i);
        integral_value = integral(@(theta) integrand(theta, rho), 0, 2*pi);
        B_values(i) = (mu0 * I * R / (4 * pi)) * integral_value;
    end

    % Plot dei risultati
    figure;
    plot(rho_values, B_values, 'b', 'LineWidth', 2);
    xlabel('\rho');
    ylabel('B_{\phi}');
    title(['Campo B_{\phi} in funzione di \rho per R=', num2str(R), ', I=', num2str(I)]);
    grid on;

    % Output dei valori di B_phi
    B_phi = B_values;
end