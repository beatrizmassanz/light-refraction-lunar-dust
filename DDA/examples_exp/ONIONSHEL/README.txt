Notes for ONIONSHEL example calculation.

Aim is for spherical shell with outer radius R_out = 0.5um
                                inner radius R_in  = 0.4um
and vacuum core.

effective radius aeff = (3V/4pi)^{1/3}
                      = [(3/4pi)*(4pi/3)*(R_out^3 - R_in^3)]^{1/3}
                      = (0.5^3-0.4^3)^{1/3}
                      = 0.39365 um

ddscat.par has
SHPAR1 = 0 to have vacuum core
SHPAR2 = 48.49 to have outer diameter = 48.49 d
SHPAR3 = 0.8 = R_in/R_out

With these values of SHPAR1, SHPAR2, SHPAR3 ddscat generates a target
with N=29136 physical dipoles in the target
in a computational volume (48d)^3 = 110592 d^3

with aeff=0.39365um, the interdipole spacing d is then determined by
Nd^3 = (4pi/3)*aeff^3 -> d = (4pi/3N)^{1/3}*aeff = 0.02062 um

we set wavelength to 0.6um

thus kd = 2*pi*d/wave = 0.21595 

material in shell is assumed to be isotropic Si3N4
refractive index filename = ../diel/Si3N4
because ONIONSHEL was intended to support study of uniaxial materials in shell,
we need to provide dielectric function for radial direction
and dielectric function for directions perpendicular to radius
[for both cases we provide dielectric function for Si3N4]

Target is oriented with BETA=0,THETA=0,PHI=0
(since ideal target is spherically symmetric, results should be insensitive
to values of BETA,THETA,PHI).

We calculate scattering in one scattering plane, for scattering angles
from 0 to 180, in increments of 5 deg.

Results: Q_ext = 6.2345
         Q_abs = 0.2362

Thus C_ext = 6.2345*pi*aeff^2 = 3.035 um^2
     C_abs = 0.2362*pi*aeff^2 = 0.115 um^2
