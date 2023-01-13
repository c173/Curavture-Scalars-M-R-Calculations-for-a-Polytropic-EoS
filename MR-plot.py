import matplotlib.pyplot as plt
import numpy as np

rho_c, Mass, Radius, stiffness, compactness = np.loadtxt('MR.out', float, usecols=(0, 1, 2, 3, 4), unpack=True)

# rho_c vs M
plt.figure(1)
plt.plot(rho_c, Mass)
plt.xscale('log')
plt.xlim(1E14, 1E15)
plt.xlabel(r'$\rho_{\rm c}$ (g cm$^{-3}$)')
plt.ylabel(r'$M/M_{\odot}$')
plt.savefig("rho_c-mass.pdf", bbox_inches='tight', pad_inches=0)

# P_c/rho_c vs compactness
plt.figure(2)
plt.plot(stiffness, compactness)
plt.xlabel(r'$P_{\rm c}/\rho_{\rm c}c^2$')
plt.ylabel(r'$2GM/Rc^2$')
plt.savefig("stiffness-compactness.pdf", bbox_inches='tight', pad_inches=0)

# Mass-Radius
plt.figure(3)
plt.plot(Radius, Mass)
plt.xlabel(r'$R$ (km)')
plt.ylabel(r'$M/M_{\odot}$')
plt.savefig('MR.pdf', bbox_inches='tight', pad_inches=0)

plt.show()
