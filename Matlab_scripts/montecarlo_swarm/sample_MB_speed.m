function v_samples = sample_MB_speed(m, N)
    % SAMPLE_MB_SPEED Sample N speed (modulus) from Maxwell-Boltzman
    % distribution for a particle coming from Sun
    %
    %   Input:
    %       m: particle mass (kg)
    %       N: number of samples
    %
    %   Output:
    %       v_samples: array with sampled speeds (m/s)

    % Boltzmann constand and Sun's photosphere temperature
    kB = 1.38e-23;   % J/K
    T = 5800;        % K

    % Adimensional density function: f(v) ∝ v^2 * exp(-v^2)
    f = @(v) (m/(2*pi*kB*T))^(3/2)*4*pi.*v.^2.*exp(-(m.*v.^2)./(2*kB*T));

    % Sampling interval
    v_min = 0;
    v_max = 30000;

    % Esteem of f_max for rejection sampling
    v_vals = linspace(v_min, v_max, 100000);
    f_vals = f(v_vals);
    f_max = max(f_vals);

    % Rejection sampling
    v_samples = zeros(1, N);
    i = 1;
    while i <= N
        v_rand = unifrnd(v_min,v_max); 
        y_rand = unifrnd(0,f_max);
        if y_rand <= f(v_rand) % Acceptance region
            v_samples(i) = v_rand;
            i = i + 1;
        end
    end

    % Plot of theoretical PDF of Maxwell-Boltzmann and sampled hystogram
    figure;
    histogram(v_samples, 100, 'Normalization', 'pdf');
    hold on;
    plot(v_vals, f(v_vals), 'r', 'LineWidth', 2);
    
    xlabel('Speed (m/s)');
    ylabel('Pdf');
    title('Maxwell-Boltzmann distribution');
    legend('Hystogram', 'Theoretical PDF');
    grid on;
end