function success_rate = particles_B_trajectory_simulation(R,rho,m,q,I,dt,N_max)
    % Function che simula N_max traiettorie casuali e verifica quante di
    % queste colpiscono il toro di raggio R e spessore rho.
    % Altri input:
    % - m: massa della particella da simulare
    % - q: carica della particella
    % - I: intensità della spira
    % - dt: intervallo di discretizzazione della traiettoria
    % - T_max: tempo massimo di calcolo della traiettoria

    % Campiono i moduli delle v0 a partire dalla Maxwell-Boltzmann
    v_samples = sample_MB_speed(m,N_max);
    intersect_count = 0; % Conta quante traiettorie intersecano il toro 
    for i = 1:N_max
        % Campionamento uniforme sulla sfera di raggio 4R della posizione
        % iniziale x0
        theta = unifrnd(0,2*pi);
        phi = unifrnd(0,pi);
        r0 = [4*R*sin(phi)*cos(theta),4*R*sin(phi)*sin(theta),4*R*cos(phi)];

        % Simulazione di un target causale distribuito uniformemente
        % all'interno del toro
        theta = unifrnd(0,2*pi);
        traj_0 = [R*cos(theta),R*sin(theta),0]-r0;
        v0 = (v_samples(i)/norm(traj_0)).*traj_0; % Velocità iniziale

        % Calcolo della traiettoria
        T_max = 8*R/v_samples(i);
        [r,shield] = calcola_traiettoria_B(r0,v0,q,m,R,rho,I,dt,T_max);
        if shield == false
            intersect_count = intersect_count + 1;      
        end
    end
    success_rate = 1-(intersect_count/N_max);
end