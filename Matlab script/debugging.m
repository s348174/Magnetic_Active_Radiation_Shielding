R = 10;
I = 1000;
point = [0, 0, 0];
[Bx, By, Bz] = calcola_campo_B(R, I, point);
fprintf('B field at origin: Bx = %.5e, By = %.5e, Bz = %.5e\n', Bx, By, Bz);
