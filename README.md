# The M.A.R.S. Project

## Introduction
This is a project to evaluate with a computational approach the feasibility, practicality and utility of an active magnetic shield for galactic cosmic rays (GCR). GCRs pose a serious health challenge for long space travels because the high energetic particles that compose them, are radioactive. Classic solution to this problem has been the deployment of passive shields, such as plates of alluminum or polymers. These shields however are heavy and tend to deteriorate over time. An active shield, which uses Lorentz's force to deviate a particle, may be an inviting solution definitely worth considering.

The idea is to use an empty torus-shaped coil with a constant current intensity to generate a magnetic field. The details for the derivation of the magnetic field in the $R^3$ space are given in the next sections.

This shield is tested against the most common types of energetic particles: protons, electrons, positrons and HZE (heavy energetic ions) of common energetic spectrum that can be find in the solar system, ranging from a few KeV up to tens of MeV. Monte Carlo methods and sample computations of trajectories are used in this section.

## Derivation of B field integral from Biot-Savart Law
The final form that we need to integrate is thus:

$$
B(x) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {z \cos(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} d \theta \ ;
$$
$$
B(y) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {z \sin(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} d \theta \ ;
$$
$$
B(z) = {\mu_0 I R \over 4 \pi} \int_0^{2 \pi} {R - x\cos(\theta) - y \sin(\theta) \over (x^2 + y^2 + z^2 + R^2 -2Rx \cos(\theta) - 2Ry \sin(\theta))^{3/2}} d \theta \ .
$$

## References
Link for Boris algorithm: https://github.com/iwhoppock/boris-algorithm?tab=readme-ov-file
