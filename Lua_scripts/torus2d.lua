-- Start a new magnetostatic planar problem
newdocument(0)
mi_probdef(0, 'meters', 'planar', 1e-8, 0, 30)

-- Parameters
R_major = 10.0       -- Major radius [m]
r_minor = 1.0        -- Minor radius [m]
r_outer = R_major + r_minor  -- Outer radius
r_inner = R_major - r_minor  -- Inner radius
t_shell = 0.5        -- Shell thickness in meters
I_total = 10000      -- Total current [A]
N_turns = 1          -- Single loop representation
I_per_turn = I_total / N_turns

-- Coil material
mi_getmaterial('Air')
mi_addmaterial('Copper', 1, 1, 0, 0, 0, 0, 0, 0, 58e6, 0)
mi_addcircprop('coil', I_total, 1)

-- Create circular ring (annulus) for hollow toroidal coil
-- Upper arcs
mi_drawarc(r_outer, 0, -r_outer, 0, 180, 1) -- Great habitat circle
mi_drawarc(r_inner, 0, -r_inner, 0, 180, 1) -- Small habitat circle
mi_drawarc(r_outer + t_shell, 0, -r_outer - t_shell, 0, 180, 1) -- Great external shell
mi_drawarc(r_inner - t_shell, 0, -r_inner + t_shell, 0, 180, 1) -- Great external shell

-- Lower arcs
mi_drawarc(-r_outer, 0, r_outer, 0, 180, 1) -- Great habitat circle
mi_drawarc(-r_inner, 0, r_inner, 0, 180, 1) -- Small habitat circle
mi_drawarc(-r_outer - t_shell, 0, r_outer + t_shell, 0, 180, 1) -- Small internal shell
mi_drawarc(-r_inner + t_shell, 0, r_inner - t_shell, 0, 180, 1) -- Small internal shell

-- Add narrow outer copper shell
-- Great external shell
mi_addblocklabel(r_outer + t_shell / 2, 0)
mi_selectlabel(r_outer + t_shell / 2, 0)
mi_setblockprop('Copper', 1, 0, 'coil', 0, 0, N_turns)
mi_clearselected()
--Small internal shell
mi_addblocklabel(r_inner - t_shell / 2, 0)
mi_selectlabel(r_inner - t_shell / 2, 0)
mi_setblockprop('Copper', 1, 0, 'coil', 0, 0, N_turns)
mi_clearselected()

-- Add air-filled core (inside shell)
mi_addblocklabel(R_major, 0)
mi_selectlabel(R_major, 0)
mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
mi_clearselected()

-- Add air-filled external region
mi_addblocklabel(2 * R_major, 0)
mi_selectlabel(2 * R_major, 0)
mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
mi_clearselected()

-- Add air-filled center
mi_addblocklabel(0, 0)
mi_selectlabel(0, 0)
mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
mi_clearselected()

-- Draw symmetric circular boundary around coil center
boundary_radius = 5 * R_major
segments = 180
theta_step = 360 / segments

for i = 0, segments - 1 do
    theta1 = i * theta_step
    theta2 = (i + 1) * theta_step
    x1 = boundary_radius * cos(rad(theta1))
    y1 = boundary_radius * sin(rad(theta1))
    x2 = boundary_radius * cos(rad(theta2))
    y2 = boundary_radius * sin(rad(theta2))
    mi_addnode(x1, y1)
    mi_addnode(x2, y2)
    mi_addarc(x1, y1, x2, y2, theta_step, 1)
end

-- Add and apply zero potential boundary condition
mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0)

for i = 0, segments - 1 do
    theta1 = i * theta_step + theta_step / 2
    x = boundary_radius * cos(rad(theta1))
    y = boundary_radius * sin(rad(theta1))
    mi_selectarcsegment(x, y)
end

mi_setarcsegmentprop(theta_step, 'A=0', 0, 0)
mi_clearselected()

-- Save, run, and view the solution
mi_saveas('toroidal_biot_savart.fem')
mi_analyze()
mi_loadsolution()

exit()
