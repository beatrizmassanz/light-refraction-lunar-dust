problem parameters

x = (2*pi*a/lambda) = 5
lambda = 0.5um
a_eff = a = x*lambda/(2*pi)
          = 5*0.5/(2*pi)
          = 0.39789um
m = 0.96+1.01i (refractive index of Au at lambda=0.5um)

center of sphere is at (x_tf,y_tf,z_tf)=(0,0,0)

nearfield calculation runs from x_tf=-1.5*a to x_tf=+1.5a
                                y_tf=-1.5*a to y_tf=+1.5a
                                z_tf=-1.5*a to z_tf=+1.5a
or x_tf = -0.59684um to x_tf = +0.59684um
we will use callreadE to evaluate E along a line with y_tf=z_tf=0 
