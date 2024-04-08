Notes for 0_CUBE example calculation
Aim is for target to be 0.16um x 0.16um x 0.16um
                       V= .00419 um^3
ddscat calls for target option RCTGLPRSM with SHPAR = 16 16 16
or
L_x = 16d, L_y = 16d, L_z = 16d

target has N=16x16x16 = 4096 dipoles

aeff=(3V/4pi)^{1/3} = 0.2um

all dipoles have composition corresponding to dielectric function given
by ../diel/Au_evap   (evaporated Au)

Because the file Au_evap assumes wavelengths to be in micron, microns
are used for specifying the wavelength and target size

0.56 = vacuum wavelength (microns)

0.2 = effective radius (microns)

with this effective radius, 

   d^3 = V/N =

   d = (.00419/4096)^{1/3} = .01um

  L_x = 16d = 0.16 micron
  L_y = 16d = 0.16 micron
  L_z = 16d = 0.16 micron

Target is oriented with BETA=0, THETA=0, PHI=0:
radiation is incident along x-axis in target frame.

Scattering is calculated for plane with phi = 0 (x-y plane in TF)
                and also for plane with phi = 90 deg (x-z plane in TF)

