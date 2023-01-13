# Curvature-Scalars-and-M-R-Calculations-for-a-Polytropic-EoS
This is a python code using arrays and RK4 method to calculate the mass-radius relations and curvature scalars of a given central density
Use rk4.py to calculate the M-R Relations and Curvature Scalars&Mass, Radius, Pressure of the star for a given central density.
You need to manually change the cutoff pressure to prevent it to get smaller than zero, adjust it as you change the central density.
Weyl Scalar blows up the origin since the difference of the terms inside it are too close to eachother, it will be fixed by expanding it to taylor series around the origin in the later versions.
