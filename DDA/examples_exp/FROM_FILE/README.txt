Notes for FROM_FILE example calculation

NB: this is the same physical problem as for the example in RCTGLPRSM,
    except that the target geometry is input from a file
    the numerical results for scattering should be identical

ddscat.par calls for target option FROM_FILE, meaning that code will look
for file "shape.dat" containing locations of target dipoles
For this example the file "shape.dat" corresponds to a rectangular
prism of dimensions 32d x 64d x 64d
containing N = 32x64x64 = 131072 dipoles
all dipoles have composition "1" for x, y, and z axes.
the dielectric function for composiition 1 is taken to be that
of evaporated Au, from file ../diel/Au_evap

calculation parameters (unit of length is taken to be microns):

0.50 = vacuum wavelength (micron)

0.49237 = aeff = effective radius (micron)

with this effective radius, 

   d^3 = V/N = (4*pi/3)*aeff^3 / N
   d = [(4*pi/3)/N]^{1/3} * aeff
     = [(4*pi/3)/131072]^{1/3} * 0.49237 = 0.01562 micron

  L_x = 32d = 0.5000 micron
  L_y = 64d = 1.0000 micron
  L_z = 64d = 1.0000 micron

Target is oriented with BETA=0, THETA=0, PHI=0:
radiation is incident along x-axis in target frame.

Scattering is calculated for plane with phi = 0 (x-y plane in TF)
                and also for plane with phi = 90 deg (x-z plane in TF).


