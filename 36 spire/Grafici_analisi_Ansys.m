clc
clear

% Leggiamo i file di testo esportati da Ansys (i file sono esportati con la virgola e matlab non la accetta)
nome_fileX = 'risultati_Ansys_asseX.txt';
data_rawX = readmatrix(nome_fileX, 'OutputType', 'string');
data_dotsX = strrep(data_rawX, ',', '.');
data_numX = str2double(data_dotsX);

nome_fileY = 'risultati_Ansys_asseY.txt';
data_rawY = readmatrix(nome_fileY, 'OutputType', 'string');
data_dotsY = strrep(data_rawY, ',', '.');
data_numY = str2double(data_dotsY);

nome_fileZ = 'risultati_Ansys_asseZ.txt'; 
data_rawZ = readmatrix(nome_fileZ, 'OutputType', 'string');
data_dotsZ = strrep(data_rawZ, ',', '.');
data_numZ = str2double(data_dotsZ);

% Definiamo le coordinate dei nodi utilizzati da Ansys
X = data_numX(:,2);
Y = data_numY(:,3);
Z = data_numZ(:,4);

% Componenti del campo magnetico
Bx = data_numX(:,end);
By = data_numY(:,end);
Bz = data_numZ(:,end);

B_mag = sqrt(Bx.^2 + By.^2 + Bz.^2);

% Definire 
figure;
scatter3(X,Y,Z,1, B_mag,'filled')
colormap(jet);
colorbar;
axis equal;
title('Campo Magnetico da Ansys (Modello Spira Singola)');
view(3);

F = scatteredInterpolant(X, Y, Z, B_mag, 'linear', 'none');
[x_query, z_query] = meshgrid(-15:0.2:15, -10:0.2:10);
y_query = zeros(size(x_query)); %Y = 0
B_sezione = F(x_query, y_query, z_query);

figure;
contourf(x_query, z_query, B_sezione, 100, 'LineColor', 'none');
colorbar;
colormap(jet);
axis equal;
title('Sezione 2D dal Modello Ansys (Piano Y=0)');
xlabel('X (m)'); 
ylabel('Z (m)');