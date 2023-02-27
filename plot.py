import numpy as np
import matplotlib.pyplot as plt

Radius, rho, P, Mass, moment_of_inertia = np.loadtxt('radial.out', float, usecols=(0, 1, 2, 3, 4), unpack=True)
Ricci_scalar, Weyl_scalar, fully_contracted_Ricci, Kretschmann_scalar, compactness = np.loadtxt('curvatures.out', float,
                                                                                                usecols=(0, 1, 2, 3, 4),
                                                                                                unpack=True)

# Plots
# Density
plt.figure(1)
plt.plot(Radius, rho)
plt.yscale('log')
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'$\rho$ (10$^{15}$ g cm$^{-3}$)')
plt.savefig('density.pdf', bbox_inches='tight', pad_inches=0)

# Pressure
plt.figure(2)
plt.plot(Radius, P)
plt.yscale('log')
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'$P$ (10$^{35}$ dyne cm$^{-2}$)')
plt.savefig('pressure.pdf', bbox_inches='tight', pad_inches=0)

# Mass
plt.figure(3)
plt.plot(Radius, Mass)
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'$M/M_{\odot}$')
plt.savefig('mass.pdf', bbox_inches='tight', pad_inches=0)

# Moment of Inertia
plt.figure(4)
plt.plot(Radius, moment_of_inertia)
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'Moment of Inertia (I)')
plt.show()

# -----------------------------------------------


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111, label="1")
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()

l1, = ax1.plot(Radius, 1e12 * Ricci_scalar, color='red', label=r'$\mathcal{R}$')
ax1.set_xlabel(r'$r$ (km)')
ax1.tick_params(axis='x', colors="black")
ax1.set_ylabel(r'Curvature $(10^{-12})$ cm$^{-2}$')

l2, = ax2.plot(Radius, 1e12 * fully_contracted_Ricci, color='green', label=r'$\mathcal{J}$')
ax2.set_yticks([])
l3, = ax3.plot(Radius, 1e12 * Kretschmann_scalar, color='blue', label=r'$\mathcal{K}$')
ax3.set_yticks([])

l4, = ax4.plot(Radius, 1e12 * Weyl_scalar, color='pink', label=r'$\mathcal{W}$')
ax4.set_yticks([])

l5, = ax5.plot(Radius, compactness, color='black', label=r'$\eta$')
ax5.set_ylabel(r'Compactness $(\eta)$')
ax5.yaxis.tick_right()

plt.legend((l1, l2, l3, l4, l5), [r'$\mathcal{R}$', r'$\mathcal{J}$', r'$\mathcal{K}$', r'$\mathcal{W}$', r'$\eta$'],
           loc='lower left')
plt.show()
fig.savefig('curvatures.pdf')
