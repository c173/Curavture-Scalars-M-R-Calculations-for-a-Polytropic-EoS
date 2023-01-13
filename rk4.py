import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Use flag=0 for radial mode, flag=1 for M-R mode
flag = 1
# -----------------------------------------------

# constants
pi = np.pi
m_sun = 1.989e33
km = 1e5  # cm
G = 6.67430e-8  # Gravitational constant cm^3/(g*s)^
c = 2.998e+10  # cm/s
k = 8 * pi * G / (c ** 4)  # cm^7/(g*s^6)
# -----------------------------------------------

# parameters
gamma = 5 / 3
K = 1e10
rho_c = 2e17  # Density central g/cm^3


# -----------------------------------------------


# derivative functions
def der_m(r, rho):
    dmdr = 4 * pi * r ** 2 * rho
    return dmdr


def der_P(r, rho, m, P):
    cor1 = 1 + P / (rho * c ** 2)
    cor2 = 1 + 4 * pi * r ** 3 * P / (m * c ** 2)
    cor3 = 1 - 2 * G * m / (r * c ** 2)
    corr = cor1 * cor2 / cor3
    dPdr = - G * m * rho * corr / r ** 2
    return dPdr


# -----------------------------------------------


def radial(rho_c):
    # Defining the arrays and the r grid.
    r = np.zeros(10**5)
    rho = np.zeros(len(r))
    m = np.zeros(len(r))
    P = np.zeros(len(r))
    moment_of_inertia = np.zeros(len(r))

    m_k1 = np.zeros(len(r))
    m_k2 = np.zeros(len(r))
    m_k3 = np.zeros(len(r))
    m_k4 = np.zeros(len(r))

    p_k1 = np.zeros(len(r))
    p_k2 = np.zeros(len(r))
    p_k3 = np.zeros(len(r))
    p_k4 = np.zeros(len(r))

    # P1 = np.zeros(len(r))
    # rho1 = np.zeros(len(r))
    # density = np.zeros(len(r))
    # pressure = np.zeros(len(r))

    # -----------------------------------------------

    #Interpolation -in work-
    #density, pressure = np.loadtxt('AP4_cgs.dat', float, usecols=(0, 1), unpack=True)

    #rho_min = density[4]  # listedeki 5. değer
    #P_min = pressure[4]
    #rho_max = density[-1]  # listedeki sondeğer
    #P_max = pressure[-1]

    #P_interp = interp1d(density, pressure, kind='linear')  # kind='cubic'
    #rho_interp = interp1d(pressure, density, kind='linear')

    #P_c = 10 ** P_interp(rho_c)
    P_c = K * rho_c ** gamma  # Poly-tropic EoS

    # Initial values
    # center
    dr = 1e3  # cm
    r[0] = dr
    rho[0] = rho_c
    m[0] = 4 * pi * dr ** 3 * rho_c / 3
    P[0] = P_c

    # first sphere
    r[1] = dr
    rho[1] = rho_c
    P[1] = P_c
    m[1] = 4 * pi * dr ** 3 * rho_c / 3

    moment_of_inertia[0] = m[0] * r[0] ** 2
    moment_of_inertia[1] = m[1] * r[1] ** 2
    # -----------------------------------------------

    # RK4 with numpy arrays
    for i in range(1, len(r) - 1):

        if P[i] < 10**30:
            break
        else:
            m_k1[i] = der_m(r[i], rho[i]) * dr
            m_k2[i] = der_m(r[i] + 0.5 * dr, rho[i] * m_k1[i] * 0.5) * dr
            m_k3[i] = der_m(r[i] + 0.5 * dr, rho[i] * m_k2[i] * 0.5) * dr
            m_k4[i] = der_m(r[i] + 0.5 * dr, rho[i] * m_k3[i] * 0.5) * dr
            m[i + 1] = m[i] + (1 / 6) * (m_k1[i] + 2 * m_k2[i] + 2 * m_k3[i] + m_k4[i])

            p_k1[i] = der_P(r[i], rho[i], m[i], P[i]) * dr
            p_k2[i] = der_P(r[i] + 0.5 * dr, rho[i] + dr * p_k1[i] * 0.5, m[i] + dr * m_k1[i] * 0.5,
                            P[i] + dr * p_k1[i] * 0.5) * dr
            p_k3[i] = der_P(r[i] + 0.5 * dr, rho[i] + dr * p_k2[i] * 0.5, m[i] + dr * m_k2[i] * 0.5,
                            P[i] + dr * p_k2[i] * 0.5) * dr
            p_k4[i] = der_P(r[i] + 0.5 * dr, rho[i] + dr * p_k3[i], m[i] + dr * m_k3[i], P[i] + dr * p_k3[i]) * dr
            P[i + 1] = P[i] + (1 / 6) * (p_k1[i] + 2 * p_k2[i] + 2 * p_k3[i] + p_k4[i])

            moment_of_inertia[i + 1] = moment_of_inertia[i] + (8 * pi * rho[i] * r[i] ** 4) * dr
            m[i + 1] = m[i] + der_m(r[i], rho[i]) * dr
            P[i + 1] = P[i] + der_P(r[i], rho[i], m[i], P[i]) * dr

            # Equation of State
            # rho[i+1] = rho_interp(P[i+1])
            rho[i + 1] = (P[i+1]/K)**(1/gamma)

            r[i + 1] = r[i] + dr

            # Adaptive Step Size
            H_m = m[i + 1] / np.abs(der_m(r[i], rho[i]))
            H_p = P[i + 1] / np.abs(der_P(r[i], rho[i], P[i], m[i]))
            H = (H_m * H_p) / (H_m + H_p)
            dr = max(0.01 * H, 10)  # do not allow dr to be smaller than 10 cm
    # -----------------------------------------------

    # Cutting off the excess values from the array.
    P = P[P != 0]
    m = m[m != 0]
    moment_of_inertia = moment_of_inertia[moment_of_inertia != 0]
    rho = rho[rho != 0]
    r = r[r != 0]
    # -----------------------------------------------

    # Results
    Mass = m / m_sun
    Radius = r / km
    stiffness = P_c / (rho_c * c ** 2)
    compactness = 2 * G * m / (r * c ** 2)
    Ricci_scalar = (k * (rho * c ** 2 - 3 * P))
    Weyl_scalar = ((4 / 3) * (((6 * G * m) / ((c ** 2) * (r ** 3))) - (k * rho * c ** 2)) ** 2) ** 0.5
    fully_contracted_Ricci = ((k ** 2) * (((rho * c ** 2) ** 2) + 3 * P ** 2)) ** 0.5
    Kretschmann_scalar = ((k ** 2) * (3 * (rho * c ** 2) ** 2 + 3 * P ** 2 + 2 * P * rho * c ** 2) - k * (
            16 * G * m / ((r ** 3) * (c ** 2))) * (rho * c ** 2) + (48 *
                                                                    (G ** 2) * m ** 2) / ((r ** 6) * (c ** 4))) ** 0.5
    test = (Weyl_scalar ** 2 + 2 * fully_contracted_Ricci ** 2 - (1 / 3) * Ricci_scalar ** 2) ** 0.5

    # Writing the data to the file

    if flag == 0:
        f = open("radial.out", 'a')  # append the data to the file # please delete the file before new start
        for i in range(0, len(r)):
            data = [Radius[i], rho[i] / 1e15, P[i] / 1E35, Mass[i], moment_of_inertia[i]]
            np.savetxt(f, np.column_stack(data), delimiter=" ", fmt='%1.4e')
        f.close()

        f = open("curvatures.out", 'a')  # append the data to the file # please delete the file before new start
        for i in range(0, len(r)):
            data = [Ricci_scalar[i], Weyl_scalar[i], fully_contracted_Ricci[i], Kretschmann_scalar[i], compactness[i]]
            np.savetxt(f, np.column_stack(data), delimiter=" ", fmt='%1.4e')
        f.close()
    # -----------------------------------------------

    return Mass, Radius, stiffness, compactness, P
    # -----------------------------------------------


Mass, Radius, stiffness, compactness, P = radial(rho_c)

if flag == 0:
    print('A neutron star with central energy density 2e17 g/cm^3')
    print('Mass is:', Mass[-1])
    print('Radius is:', Radius[-1])
    print('Compactness is:', compactness[-1])
    print('Stiffness is:', stiffness)

if flag == 1:
    rho_central = np.logspace(14, 16, num=100, endpoint=True, base=10)
    f = open("MR.out", 'a')
    for rho_c in rho_central:
        Mass, Radius, stiffness, compactness, P = radial(rho_c)
        dat = [rho_c, Mass[-1], Radius[-1], stiffness, compactness[-1]]
        print(dat)
        np.savetxt(f, np.column_stack(dat), delimiter=" ", fmt='%1.4e')

    f.close()
