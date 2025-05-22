%% Run Lua script via wine
function lua2femm(filename)
    % Add path to mfiles
    addpath("/home/alberto/.wine/drive_c/femm42/mfiles");

    % Run Lua script in FEMM
    callfemmfile(filename);

    % Read and memorize data
    data = readmatrix('field_export.dat');
    x = data(:,1); 
    y = data(:,2); 
    Bx = data(:,3); 
    By = data(:,4);
    %Bz = data(:,4);

    % Compute B magnitude
    Bmag = sqrt(Bx.^2 + By.^2);
    
    % Determine grid size
    x_unique = unique(x);
    y_unique = unique(y);
    
    % Reshape to 2D grids
    [X, Y] = meshgrid(x_unique, y_unique);
    Bmag_grid = reshape(Bmag, length(y_unique), length(x_unique));
    % Surface plot
    figure;
    surf(X, Y, Bmag_grid);
    shading interp;
    colorbar;
    title('|B| magnitude in x-y plane');
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('|B| (T)');
    axis tight;
    view(2); % Optional: top view
end