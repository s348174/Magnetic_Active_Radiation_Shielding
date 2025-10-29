function v_samples = sample_MB_speed(m, N)
    % SAMPLE_MB_SPEED Genera N velocità (modulo) secondo la distribuzione
    % di Maxwell-Boltzman per una particella di massa m prodotta dal Sole.
    %
    %   Input:
    %       m: massa della particella (kg)
    %       N: numero di campioni da generare
    %
    %   Output:
    %       v_samples: array con i valori della velocità in m/s

    % Costante di Boltzmann e temperatura della fotosfera solare
    kB = 1.38e-23;   % J/K
    T = 5800;        % K

    % Fattore di scala per riportare in unità fisiche
    % scale = sqrt(kB * T / m);  % [m/s]

    % Funzione di densità adimensionale: f(v) ∝ v^2 * exp(-v^2)
    f = @(v) (m/(2*pi*kB*T))^(3/2)*4*pi.*v.^2.*exp(-(m.*v.^2)./(2*kB*T));

    % Intervallo di campionamento
    v_min = 0;
    v_max = 30000;

    % Stima di f_max per il rejection sampling
    v_vals = linspace(v_min, v_max, 100000);
    f_vals = f(v_vals);
    f_max = max(f_vals);

    % Rejection sampling
    v_samples = zeros(1, N);
    i = 1;
    while i <= N
        v_rand = unifrnd(v_min,v_max); 
        y_rand = unifrnd(0,f_max);
        if y_rand <= f(v_rand) % Regione di accettazione
            v_samples(i) = v_rand;
            i = i + 1;
        end
    end

    % Conversione in unità fisiche (m/s)
    % v_samples = samples * scale;

    % Plot della PDF teorica della Maxwell-Boltzmann e dell'istogramma
    figure;
    histogram(v_samples, 100, 'Normalization', 'pdf');
    hold on;
    plot(v_vals, f(v_vals), 'r', 'LineWidth', 2);
    
    xlabel('Velocità (m/s)');
    ylabel('Pdf');
    title('Distribuzione di Maxwell-Boltzmann');
    legend('Istogramma campioni', 'PDF teorica');
    grid on;
end