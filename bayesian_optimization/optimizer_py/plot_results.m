function plot_results()
    % Load results from .csv file
    data = readtable("configurations/best_configuration_K4_42.csv", 'Delimiter', ',');

    % Extract columns
    radius = data.R;
    centersX = data.cX;
    centersY = data.cY;
    centersZ = data.cZ;
    normalsX = data.nX;
    normalsY = data.nY;
    normalsZ = data.nZ;

    % Plot coils
    for i = 1:length(radius)
        % Plot planes with normals
        quiver3(centersX(i), centersY(i), centersZ(i), normalsX(i), normalsY(i), normalsZ(i), 'r');
        hold on;
        % Plot coil centers
        plot3(centersX(i), centersY(i), centersZ(i), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        % Plot coil circles
        [X, Y, Z] = sphere(20);
        X = radius(i) * X + centersX(i);
        Y = radius(i) * Y + centersY(i);
        Z = radius(i) * Z + centersZ(i);
        surf(X, Y, Z, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end

plot_results();