bx = 'bx_ansys.dbl'; % Inserisci il nome esatto del tuo file
file_bz = 'bz_ansys.dbl'; % Inserisci il nome esatto del tuo file

% Dimensioni della griglia (DEVI SAPERE QUESTE DAL TUO SOFTWARE)
% Esempio: se l'analisi ha 200 punti in raggio e 400 in altezza
nr = 401; 
nz = 401; 

% --- LETTURA DATI BINARI ---
% Apertura e lettura BX
f1 = fopen(file_bx, 'rb');
dati_bx = fread(f1, 'double');
fclose(f1);

% Apertura e lettura BZ
f2 = fopen(file_bz, 'rb');
dati_bz = fread(f2, 'double');
fclose(f2);

% --- RICOSTRUZIONE MATRICI ---
% Se i dati sono salvati per colonne (standard Fortran/Ansys)
Bx = reshape(dati_bz, nr, nz);
Bz = reshape(dati_bx, nr, nz);

% Calcolo modulo del campo (Intensità)
B_mag = sqrt(Bx.^2 + Bz.^2);

% Definizione delle coordinate (Esempio: da 0 a 10 metri)
r = linspace(0, 20, nr);
z = linspace(0, 20, nz);
[R, Z] = meshgrid(r, z); % Crea la griglia per il plot

% --- PLOT ---
figure('Color', 'k'); % Sfondo nero per risaltare come nell'immagine
hold on;

% 1. Intensità del campo con contourf
% Usiamo 100 livelli per una sfumatura fluida
[~, hContour] = contourf(R, Z, B_mag, 100, 'LineColor', 'none');
shading interp;

% 2. Linee di campo con streamslice
% Nota: streamslice vuole le matrici coordinate e le componenti
st = streamslice(R, Z, Bz, Bx, 2); 
set(st, 'Color', 'w', 'LineWidth', 1.0); % Linee bianche

% --- ESTETICA ---
colormap(hot); % 'hot' o 'magma' ricordano molto i colori della tua cupola
c = colorbar;
c.Color = 'w'; % Colore del testo della colorbar bianco
c.Label.String = 'Intensità B (Tesla)';
xlabel('Raggio (r)');
ylabel('Altezza (z)');
title('Sezione Assial-simmetrica: Campo Post-Interazione', 'Color', 'w');
axis equal tight;
grid off;
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');

% Limita la scala se necessario per vedere i dettagli fuori dalla cupola
% clim([0 0.5]);