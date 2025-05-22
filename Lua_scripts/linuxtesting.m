% Create a Lua script
fid = fopen('coiltest.lua', 'wt');
fprintf(fid, 'newdocument(0)\n');
fprintf(fid, 'mi_probdef(0, "meters", "planar", 1e-8, 0, 30)\n');
fprintf(fid, 'mi_saveas("coiltest.fem")\n');
fclose(fid);

% Run it via Wine
callfemmfile('coiltest.lua');