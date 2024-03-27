Notes for SPH_ANI_N example

Calculate scattering and absorption by an random cluster of 64 spheres
(32 silicate and 32 graphite).  The cluster is a random realization
generated using the "BAM2" algorithm (Ballistic Aggregation, 2
Migrations) described in Shen, Draine, & Johnson (2008).

Graphite is an anisotropic material; the graphite spheres are assigned
random orientations.

The sphere locations, compositions, and orientations are specified in the
file BAM2.64.1.50.targ

The amorphous silicate dielectric function is read from
"../diel/astrosil" taken from Draine (2003).
The dielectric function for graphite, E || c, is read from
"../diel/graphite_E_para_c" taken from Draine (2003).
The dielectric function for graphite, E perp c, is read from
"../diel/graphite_E_perp_c" taken from Draine (2003).

The dielectric files specify wavelength in microns, therefore ddscat.par
specifies wavelength and particle size in micros.

The ambient medium has refractive index = 1 (i.e., vacuum).

The effective radius of the cluster is aeff = 0.20 micron.  aeff is
the radius of an equal-volume sphere.  Given that 64 spheres are
present, each of the constitutive spheres has a radius
0.20um/(64)^{1/3} = 0.05um

Assignment of dipoles to the target is specified by the line
27.84 1 'BAM2.64.1_50.targ' = shape parameters SHPAR1, SHPAR2, filename

Here SHPAR1=27.84 specifies the extent of the structure in the x-direction
in units of d.  The peculiar number 27.84 is chosen so that each of
the constitutive spheres will contain at least 100 dipoles, with SHPAR1
increased beyond the minimum value until further increase would result
in an increase in the "computational volume".  For this example, the
total number of dipoles turns out to be N=7947, or 7947/64=124 dipoles per
sphere on average.

SHPAR2 specifies that the "target axes" a_1 and a_2 should be taken to
be along the principal axes with largest and intermediate moment of
inertia.

The dipole spacing in physical units is given by Nd^3 = (4*pi/3)aeff^3
or d = (4*pi/3N)^{1/3} aeff = 0.01616 um

ddscat.par calls for the scattering calculation to be done for both
incident polarizations, for 

9 different angles between a_1 and the "lab frame" x-axis (direction
of the incident radiation), over the range 0 < BETA < 180 deg,

6 different rotations of the
target around the a_1 axis, over the range 0 < THETA < 360 deg,

and 16 different rotations of a1 around the "lab frame" x axis,
over the range 0 < PHI < 360 deg.
for a total
of 9*6*16=864 different orientations.

Setting IWRKSC=0 suppresses writing out the scattering results for each
orientation -- only the orientational average is stored (in the file
w000r000.avg

We write out the scattering properties for 4 different scattering planes.
An accurate average over random orientations would result in the same
scattering pattern for each of the scattering planes.  The actual results
show small differences because we have averaged over a finite number of
orientations.  For example at scattering angle theta=45 deg, the values of
S_11 for the 4 scattering planes are
7.5237 for phi=0
4.3048 for phi=90
6.0209 for phi=180
4.2506 for phi=270

References:

Draine, B.T. 2003, "Scattering by Intertellar Dust Grains. I. Optical
and Ultraviolet", Astrophys. J. 598, 1017-1025

Shen, Y., Draine, B.T., & Johnson, E.T. 2008, "Modeling Porous Dust
Grains with Ballistic Aggregates I: Geometry and Optical Properties",
Astrophys. J., 689, 260-275.
