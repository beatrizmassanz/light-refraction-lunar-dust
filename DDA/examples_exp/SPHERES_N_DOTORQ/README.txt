Notes for SPHERES_N example

                        Geometry

ddscat.par is set up to calculate absorption and scattering by a "target unit
cell" (TUC) composed of a cluater of spheres, with the periodic repetition
of the TUC in the y and z directions.

The arrangement of the spheres in the TUC is specified by the file
'BAM2.16.1.targ' giving the locations and sizes of 16 spheres
In BAM2.16.1.targ the spheres all have radii a0 = 0.5*D0
where D0 is some as-yet unknown length.

The spheres do not overlap, hence the effective radius of the material in
one TUC is aeff = 16^{1/3}*(D0/2)

Inspecting BAM2.16.1.targ one finds that the target extends in the
x-direction from 
(-0.433343-0.5)*D0 = -0.933343*D0 
to 
(+1.478047+0.5)*D0 =  1.978047*D0

ddscat.par specifies SHPAR1 SHPAR2 = 24 1
SHPAR1 = maximum extent/d of the TUC in the TF x-direction
SHPAR2 = 1 for target axes a1,a2 to be taken along principal axes
           of largest moment of inertia.

Thus we expect that 24*d = (1.978047+0.933343)D0 = 2.9114*D0
Thus D0 = 24*d/2.9114 = 8.2435 d
     a0 = D0/2 = 12/2.9114 d = 4.1217 d

Thus individual spheres should each represented by approximately

N_D = (4*pi/3)*a0^3 / d^3 dipoles
N_D = (4*pi/3)*(a0/d)^3 = (4*pi/3)*(4.1217)^3 = 293 dipoles

and the cluster of 16 spheres is expected to contain approximately 
N = 16*293 = 4693 dipoles

The actual number of dipoles used to represent the cluster
is (see ddscat.log) N=4663

                Scattering problem

The spheres are assumed to have radii a0 = 0.1um
The effective radius of the cluster is aeff = 16^{1/3} * a0 = 0.25198um
The incident wavelength is lambda = 0.6 um
The refractive index of the sphere material is m=1.33+0.01i


