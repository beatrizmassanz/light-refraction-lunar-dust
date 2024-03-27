RCTGLPBC used to simulate infinite slab
  infinite slab , m=(1.50,0.02), thickness = 0.10um, wave=0.50um
  using 20x1x1 TUC (M,N)=(0,0)

a = 0.10um = slab thickness
b = a/20 = 0.005um
c = c/20 = 0.005um
V = a*b*c=0.10*0.005*0.005 um^3
aeff = (3*V/4*pi)^{1/3} = 8.4195e-3 um

shpar1 = 20  = X/d
shpar2 =  1  = Y/d
shpar3 =  1  = Z/d
shpar4 =  1  = P_y/d
shpar5 =  1  = P_z/d
shpar6 =  1  to set a1=x_TF, a2=y_TF
wave = 0.5um
allowed (M,N):
M_max = int(L_y/wave)=int(0.005/0.5)=0

a1=x_tf, THETA=40 deg: radiation incident at 40 deg from surface normal.
