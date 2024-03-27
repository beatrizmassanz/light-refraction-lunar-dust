DSKRCTPBC = rectangular brick with a disk on top

Suppose lambda = 5320 A = 532 nm
Suppose we want array of Au disks with 
    diameter 500A = 50 nm
    height 400A = 40 nm (in x_TF direction)
    center-to-center distance in y_TF direction = 800A = 80 nm
    center-to-center distance in z_TF direction = 800A = 80 nm
on top of Si3N4 glass substrate of thickness 500A (in x_TF direction)
    dipole spacing d = 50A = 5 nm

shpar1 = (disk thickness in x_TF direction)/d  = 400/50 = 8
shpar2 = (disk diameter)/d                     = 500/50 = 10
shpar3 = (brick thickness in x_TF direction)/d = 500/50 = 10
shpar4 = (brick thickness in y_TF direction)/d = 800/50 = 16
shpar5 = (brick thickness in y_TF direction)/d = 800/50 = 16
shpar6 = (periodicity in y_TF direction)/d     = 800/50 = 16
shpar7 = (periodicity in z_TF direction)/d     = 800/50 = 16
shpar8 = 1 : set a_1=x_TF, a_2=y_TF

VTUC/d^3 = pi*s1*(s2/2)**2 +s3*s4*s5 = pi*8*5*5 + (10*16*16) 
                                     = 3188.32

Then aeff = (3*VTUC/4*pi)^{1/3} = 9.130426*d = 456.521A = 45.6521nm= 0.0456521um

ideal N = 398.5. Actual N = 384
ideal x_j runs from -9.5*5 to +7.5*5 = -47.5 to +37.5 nm
      y_j runs from -7.5*5 to +7.5*5 = -37.5 to +37.5 nm
      z_j runs from -7.5*5 to +7.5*5 = -37.5 to +37.5 nm

radiation incident in lab frame with theta=60 deg relative to Si3N4 normal
k=|k|(cos(theta),-sin(theta),0)   --> THETA=60
                                      PHI=0
                                      BETA=0

ddfield.in : calculate E along line (x,4d,0) running from bottom of Si3N4 slab
             to top of Si3N4 slab, continuing within Au disk, 1d from edge
             Au disk diameter = 10d, radius = 5d
             Au disk thickness = 8d
