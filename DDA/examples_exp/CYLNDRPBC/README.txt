x=5 calculation for cylinder with |m|=1.33

shpar1=1      : cylinder length along axis
shpar3=1      : cylinder axis || y_TF
shpar4=1      : P_y/d
shpar5=0      : no repetition in z_TF direction
shpar6=4      : a_1=y_TF,a_2=z_TF

shpar2 = (cylinder diameter 2R)/d

Take wavelength = 1 unit
Cylinder radius = R
Want 2*pi*R/lambda = 5 
Hence set R=5*lambda/2*pi = 5/(2*pi) units

take shpar2 = 64.499
The estimated number of dipoles in this slice would be N=pi*(64.499/2)^2=3267.
The actual number turns out to be N=3260.
Thus the "radius" of this slice R=sqrt(3260/pi)d=32.213d
Thus d=R/sqrt(N/pi)

Now need to determine aeff
V=N*d^3
aeff = (3*V/4*pi)^{1/3} = (3N/4pi)^{1/3} d
     = (3N/4pi)^{1/3} * R*sqrt(pi/N)
     = (3/4pi)^{1/3} * sqrt(pi) * R / N^{1/6}
     = 0.22723 units 

------------ Target Orientation Relative to Incident Radiation ------

The cylinder symmetry axis is in the y_TF direction.
We choose to set a_1 = y_TF = cylinder axis
                 a_2 = z_TF

The target orientation in the Lab Frame is set to

   BETA = 0 (no rotation arount a_1)
   THETA = 60 deg (radiation incident at 60 deg from cylinder axis a_1=y_TF)
   PHI = 0 (no rotation of target around a_1)
                                 
The sample calculation is thus for radiation incident at an angle
theta=30 deg away from normal to the cylinder symmetry axis
(i.e., at an angle 60 deg relative to the symmetry axis).

The scattered radiation in order M=0 will be in a cone at an angle
alpha=60 deg away from the cylinder axis.

