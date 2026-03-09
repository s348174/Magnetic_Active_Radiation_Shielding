clc
clear
close all

%% Addpath to mfiles in FEMM 4.2 installation (to be changed by every user)
addpath("C:\Users\lukea\Downloads\femm42\mfiles");

openfemm;

%% Start a new magnetostatic axisymmetric problem

newdocument(0)

mi_probdef(0, 'meters', 'planar', 1e-8, 0, 30)



%% Parameters

R_major = 10.0;       % Major radius [m]

r_minor = 1.0;        % Minor radius [m]

r_outer = R_major + r_minor; % Outer radius

r_inner = R_major + r_minor; % Inner radius

t_shell = 0.2;       % Shell thickness in meters

I_total = 10000;     % Total current [A]

N_turns = 1;         % For Biot-Savart law, single loop representation

I_per_turn = I_total / N_turns;


% Coil material

mi_getmaterial('Air');

mi_addmaterial('Copper', 1, 1, 0, 0, 0, 0, 0, 0, 58e6, 0);

mi_addcircprop('coil', I_total, 1);




%% Create circular ring (annulus) for hollow toroidal coil

% Draw upper arc

mi_drawarc(R_major + r_minor, 0, -R_major - r_minor, 0, 180, 1); % External circle

mi_drawarc(R_major - r_minor, 0, -R_major + r_minor, 0, 180, 1); % Internal circle

% Draw lower arc (completing the circle)

mi_drawarc(-R_major - r_minor, 0, R_major + r_minor, 0, 180, 1); % External circle

mi_drawarc(-R_major + r_minor, 0, R_major - r_minor, 0, 180, 1); % Internal circle


% Add narrow outer copper shell

mi_addblocklabel(r_outer - t_shell/2, 0);

mi_selectlabel(r_outer - t_shell/2, 0);

mi_setblockprop('Copper', 1, 0, 'coil', 0, 0, N_turns);

mi_clearselected();

% Add air-filled core (no current)

mi_addblocklabel(R_major, 0);

mi_selectlabel(R_major, 0);

mi_setblockprop('Air', 1, 0, '', 0, 0, 0);

mi_clearselected();

% Add air-filled external

mi_addblocklabel(2*R_major, 0);

mi_selectlabel(2*R_major, 0);

mi_setblockprop('Air', 1, 0, '', 0, 0, 0);

mi_clearselected();

% Add air-filled center

mi_addblocklabel(0, 0);

mi_selectlabel(0, 0);

mi_setblockprop('Air', 1, 0, '', 0, 0, 0);

mi_clearselected();


% Assign copper material and current to the torus

%mi_addblocklabel(R_major, 0);

%mi_selectlabel(R_major, 0);

%mi_setblockprop('Copper', 1, 0, 'coil', 0, 0, N_turns);

%mi_clearselected();



%% Draw symmetric circular boundary around coil center

boundary_radius = 5*R_major;

segments = 180;

theta_step = 360 / segments;

% Add boundary nodes and arcs

for i = 0:segments-1

    theta1 = i * theta_step;

    theta2 = (i + 1) * theta_step;

    x1 = boundary_radius * cos(theta1 * pi / 180);

    y1 = boundary_radius * sin(theta1 * pi / 180);

    x2 = boundary_radius * cos(theta2 * pi / 180);

    y2 = boundary_radius * sin(theta2 * pi / 180);

    mi_addnode(x1, y1);

    mi_addnode(x2, y2);

    mi_addarc(x1, y1, x2, y2, theta_step, 1);

end



%% Add and apply zero potential boundary condition

mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0);

for i = 0:segments-1

    theta1 = i * theta_step + theta_step / 2;

    x = boundary_radius * cos(theta1 * pi / 180);

    y = boundary_radius * sin(theta1 * pi / 180);

    mi_selectarcsegment(x, y);

end

mi_setarcsegmentprop(theta_step, 'A=0', 0, 0);

mi_clearselected();



%% Save, run, and view the solution

mi_saveas('toroidal_biot_savart.fem');

mi_analyze();

mi_loadsolution();

