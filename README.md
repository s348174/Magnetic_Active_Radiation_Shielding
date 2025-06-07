# Documentazione

## Ricavazione del campo dalla legge di Biot-Savart
La forma finale del campo è quindi:

$$
B(x) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {z \cos(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} \ ;
$$
$$
B(y) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {z \sin(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} \ ;
$$
$$
B(z) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {R - x\cos(\theta) - y \sin(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} \ .
$$

## Approssimazione del toroide con un insieme di spire

Per mostrare la convergenza è sufficiente quindi mostrare che:
$$
\int_0^{2 \pi} \int_0^{2 \pi} {(R - \rho \cos(\alpha) ) (z - \rho \cos(\alpha) ) \cos(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}}
$$
