%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEFINE CLASS FOR A CONDUCTIVE TORUS DEFINED AD MULTIPLE PARALLEL FILAMENTS FORMING A COIL %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Torus
    properties
        % Input properties
        R;               % External radius
        rho;             % Internal radius
        I;               % Current intensity
        N_coils;         % Number of coils for discrete calculation
        
        % Properties calculated from input
        coils_centers    % Array of size N_wires containing the center of each coil
        coils_radii      % Array containing the radius of each coil
    end
    methods
        function obj = Torus(R,rho,I,N_coils) % Class constructor
            obj.R = R;
            obj.rho = rho;
            obj.I = I/N_coils;
            obj.N_coils = N_coils;
            
            angles = (linspace(0,2*pi,obj.N_coils+1))';
            angles(end) = [];
            obj.coils_centers = zeros(obj.N_coils,3);
            obj.coils_centers(:,3) = rho.*sin(angles);
            obj.coils_radii = R.*ones(obj.N_coils,1) - rho.*cos(angles);
            %disp(obj.coils_radii);
            %disp(obj.coils_centers);
        end
        function B_field = get_point_field(obj, point) % Function that calculates the field at given point
            coils_fields = zeros(obj.N_coils,3);
            for i = 1:obj.N_coils
                r = point-obj.coils_centers(i,:);
                %fprintf('Point:');
                %disp(r);
                %fprintf("Radius:");
                %disp(obj.coils_radii(i));
                [Bx, By, Bz] = calcola_campo_B(obj.coils_radii(i), obj.I, r, 0);
                %fprintf("Integration result:");
                %disp([Bx, By, Bz]);
                coils_fields(i,:) = [Bx, By, Bz];
                %fprintf("Single coil field:");
                %disp(coils_fields(i,:));
            end
            B_field = sum(coils_fields);
            %fprintf("Final field:");
            %disp(B_field);
        end
    end
end