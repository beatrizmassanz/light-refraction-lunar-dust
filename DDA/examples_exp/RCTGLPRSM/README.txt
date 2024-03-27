Notes for RCTGLPRSM example calculation
Aim is for target to be 0.25um x 0.5um x 0.5um
                       V= .0625 um^3
ddscat calls for target option RCTGLPRSM with SHPAR = 16 32 32
or
L_x = 16d, L_y = 32d, L_z = 32d

target has N=16x32x32 = 16384 dipoles

aeff=(3V/4pi)^{1/3} = 0.246186um

all dipoles have composition corresponding to dielectric function given
by ../diel/Au_evap   (evaporated Au)

Because the file Au_evap assumes wavelengths to be in micron, microns
are used for specifying the wavelength and target size

0.50 = vacuum wavelength (microns)

0.246186 = effective radius (microns)

with this effective radius, 

   d^3 = V/N =

   d = (.0625/16384)^{1/3} = .015625um

  L_x = 16d = 0.125 micron
  L_y = 32d = 0.25 micron
  L_z = 32d = 0.25 micron

Target is oriented with BETA=0, THETA=0, PHI=0:
radiation is incident along x-axis in target frame.

Scattering is calculated for plane with phi = 0 (x-y plane in TF)
                and also for plane with phi = 90 deg (x-z plane in TF)

