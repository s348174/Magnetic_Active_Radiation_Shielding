%% Run Lua script via wine
function lua2femm(filename)
    addpath("/home/alberto/.wine/drive_c/femm42/mfiles");
    callfemmfile(filename);
end