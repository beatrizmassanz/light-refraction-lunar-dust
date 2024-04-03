Notes for 0_SPHERE example.

ddscat.par is set up to calculate absorption and scattering by
a ellipsoid with diameters/d = 20 20 20  (d=interdipole separation).
this of course means a sphere of diameter = 20*d

m = 1.69+0.03i (refractive index of Astrosil at 560 nm)

effective radius = 0.1 micrometers

lambda = 0.56 micrometers  (wavelength in vacuum)

this gives
size parameter x = 2*pi*a/lambda = 1.122

1.0000 = NAMBIENT : ambient medium has refractive index = 1

2=IORTH : do scattering calculation for two incident polarizations
