Notes for ELLIPSOID example.

ddscat.par is set up to calculate absorption and scattering by
a ellipsoid with diameters/d = 48.49 48.49 48.49  (d=interdipole separation).
this of course means a sphere of diameter = 48.49*d

m = 0.96+1.01i (refractive index of Au at 500 nm)

effective radius = 5 units

lambda = 6.283185 units  (wavelength in vacuum)

this gives
size parameter x = 2*pi*a/lambda = 2*pi*2/2*pi = 5

1.0000 = NAMBIENT : ambient medium has refractive index = 1

2=IORTH : do scattering calculation for two incident polarizations
