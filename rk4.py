import numpy as np
from scipy.interpolate import interp1d

# Use flag=0 for radial mode, flag=1 for M-R mode
flag = 0
# -----------------------------------------------


# constants
pi = np.pi
m_sun = 1.989e33
km = 1e5  # cm
G = 6.67430e-8  # Gravitational constant cm^3/(g*s)^
c = 2.998e+10  # cm/s
k = 8 * pi * G / (c ** 4)  # cm^7/(g*s^6)
# -----------------------------------------------

conv_rho = m_sun / (km ** 3)
conv_P = conv_rho * c ** 2

# parameters
gamma = 5 / 3
K = 1e10
# rho_c = 2e17  # Density central g/cm^3
# -----------------------------------------------


density, pressure = np.loadtxt('./data/sly4.dat', float, usecols=(0, 1), unpack=True)

density = conv_rho * density
pressure = conv_P * pressure

rho_max = density[0]
P_max = pressure[0]
rho_min = density[-1]
P_min = pressure[-1]

P_interp = interp1d(np.log10(density), np.log10(pressure), kind='linear', fill_value="extrapolate")  # kind='cubic'
rho_interp = interp1d(np.log10(pressure), np.log10(density), kind='linear', fill_value="extrapolate")


# derivative functions
def der_phi(der_P, rho, P):
    term1 = -1 / (rho * c ** 2)
    term2 = der_P
    term3 = (1 + (P / (rho * c ** 2))) ** (-1)
    dphidr = term1 * term2 * term3
    return dphidr


def der_j(r, rho, m, P, phi, omega):
    term1 = (8 / 3) * pi * r ** 4
    term2 = (rho + (P / c ** 2))
    term3 = (1 - ((2 * G * m) / (r * c ** 2))) ** 0.5
    term4 = np.exp(-phi)
    term5 = omega
    djdr = term1 * term2 * term3 * term4 * term5
    return djdr


def der_omega(r, m, j, phi):
    term1 = G * np.exp(phi) / ((c ** 2) * (r ** 4))
    term2 = j
    term3 = (1 - ((2 * G * m) / (r * c ** 2))) ** -0.5
    return term1 * term2 * term3


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


def der_moi(r, rho, m, P, phi, omega):
    mom1 = 8 * pi * r ** 4 / 3
    mom2 = (rho + P / c ** 2)
    mom3 = omega * np.exp(-phi)
    mom4 = (1 - (2 * G * m) / (r * c ** 2)) ** 0.5
    return mom1 * mom2 * mom3 * mom4
# -----------------------------------------------


if flag == 0:
    f = open("radial.out", 'a')  # append the data to the file # please delete the file before new start
    g = open("curvatures.out", 'a')  # append the data to the file # please delete the file before new start
    rho_c = rho_max


def radial(rho_c):
    # -----------------------------------------------
    # P_c = K * rho_c ** gamma  # Poly-tropic EoS
    P_c = 10 ** P_interp(np.log10(rho_c))
    stiffness = P_c / (rho_c * c ** 2)

    # Initial values
    # center

    # rho_second = -(4*pi/3) * G * rho_c**3
    i = 0
    dr = 1e3  # cm
    r = dr
    rho = rho_c
    m = 4 * pi * dr ** 3 * rho_c / 3
    P = P_c
    j = 0
    omega = 1
    phi = 0
    moi = 0
    compactness = 0
    Ricci_scalar = (k * (rho * c ** 2 - 3 * P))
    Weyl_scalar = ((4 / 3) * (((6 * G * m) / ((c ** 2) * (r ** 3))) - (k * rho * c ** 2)) ** 2) ** 0.5
    fully_contracted_Ricci = ((k ** 2) * (((rho * c ** 2) ** 2) + 3 * P ** 2)) ** 0.5
    Kretschmann_scalar = ((k ** 2) * (3 * (rho * c ** 2) ** 2 + 3 * P ** 2 + 2 * P * rho * c ** 2) - k * (
            16 * G * m / ((r ** 3) * (c ** 2))) * (rho * c ** 2) + (48 *
                                                                    (G ** 2) * m ** 2) / (
                                  (r ** 6) * (c ** 4))) ** 0.5

    if flag == 0:
        data = [r / km, rho / 1e15, P / 1E35, m / m_sun, moi]
        data_cur = [Ricci_scalar, Weyl_scalar, fully_contracted_Ricci, Kretschmann_scalar, compactness]
        np.savetxt(f, np.column_stack(data), delimiter=" ", fmt='%1.4e')
        np.savetxt(g, np.column_stack(data_cur), delimiter=" ", fmt='%1.4e')

    # first sphere
    i = 1
    r = dr
    rho = rho_c
    P = P_c
    m = 4 * pi * dr ** 3 * rho_c / 3
    phi = 0
    j = 0
    omega = 1
    moi = 0
    compactness = 2 * G * m / (r * c ** 2)
    Ricci_scalar = (k * (rho * c ** 2 - 3 * P))
    Weyl_scalar = ((4 / 3) * (((6 * G * m) / ((c ** 2) * (r ** 3))) - (k * rho * c ** 2)) ** 2) ** 0.5
    fully_contracted_Ricci = ((k ** 2) * (((rho * c ** 2) ** 2) + 3 * P ** 2)) ** 0.5
    Kretschmann_scalar = ((k ** 2) * (3 * (rho * c ** 2) ** 2 + 3 * P ** 2 + 2 * P * rho * c ** 2) - k * (
            16 * G * m / ((r ** 3) * (c ** 2))) * (rho * c ** 2) + (48 *
                                                                    (G ** 2) * m ** 2) / (
                                  (r ** 6) * (c ** 4))) ** 0.5
    # -----------------------------------------------

    # RK4
    while P > P_min:

        if flag == 0:
            data = [r / km, rho / 1e15, P / 1E35, m / m_sun, moi/10**45]
            data_cur = [Ricci_scalar, Weyl_scalar, fully_contracted_Ricci, Kretschmann_scalar, compactness]
            np.savetxt(f, np.column_stack(data), delimiter=" ", fmt='%1.4e')
            np.savetxt(g, np.column_stack(data_cur), delimiter=" ", fmt='%1.4e')

        p_k1 = der_P(r, rho, m, P) * dr
        m_k1 = der_m(r, rho) * dr

        phi_k1 = der_phi(der_P(r, rho, m, P), rho, P) * dr
        omega_k1 = der_omega(r, m, j, phi) * dr
        j_k1 = der_j(r, rho, m, P, phi, omega) * dr
        moi_k1 = der_moi(r, rho, m, P, phi, omega) * dr

        r1 = r + 0.5 * dr
        m1 = m + 0.5 * m_k1
        P1 = P + 0.5 * p_k1

        phi1 = phi + 0.5 * phi_k1
        j1 = j + 0.5 * j_k1
        moi1 = moi + 0.5 * moi_k1
        omega1 = omega + omega_k1

        if (P1 < 0):
            P1 = P_min
            # rho1 = (P1/K)**(1/gamma)
            rho1 = 10 ** rho_interp(np.log10(P1))
            break

        rho1 = 10 ** rho_interp(np.log10(P1))
        p_k2 = der_P(r1, rho1, m1, P1) * dr
        m_k2 = der_m(r1, rho1) * dr

        phi_k2 = der_phi(der_P(r1, rho1, m1, P1), rho1, P1) * dr
        omega_k2 = der_omega(r1, m1, j1, phi1) * dr
        j_k2 = der_j(r1, rho1, m1, P1, phi1, omega1) * dr
        moi_k2 = der_moi(r1, rho1, m1, P1, phi1, omega1) * dr

        r2 = r1
        m2 = m + 0.5 * m_k2
        P2 = P + 0.5 * p_k2

        phi2 = phi + 0.5 * phi_k2
        j2 = j + 0.5 * j_k2
        moi2 = moi + 0.5 * moi_k2
        omega2 = omega + omega_k2

        if (P2 < 0):
            P2 = P_min
            # rho2 = (P2/K)**(1/gamma)
            rho2 = 10 ** rho_interp(np.log10(P2))
            break

        rho2 = 10 ** rho_interp(np.log10(P2))
        p_k3 = der_P(r2, rho2, m2, P2) * dr
        m_k3 = der_m(r2, rho2) * dr

        phi_k3 = der_phi(der_P(r2, rho2, m2, P2), rho2, P2) * dr
        omega_k3 = der_omega(r2, m2, j2, phi2) * dr
        j_k3 = der_j(r2, rho2, m2, P2, phi2, omega2) * dr
        moi_k3 = der_moi(r2, rho2, m2, P2, phi2, omega2) * dr

        r3 = r + dr
        m3 = m + m_k3
        P3 = P + p_k3

        phi3 = phi + phi_k3
        j3 = j + j_k3
        moi3 = moi + moi_k3
        omega3 = omega + omega_k3

        if (P3 < 0):
            P3 = P_min
            # rho3 = (P3/K)**(1/gamma)
            rho3 = 10 ** rho_interp(np.log10(P3))
            break

        rho3 = 10 ** rho_interp(np.log10(P3))
        p_k4 = der_P(r3, rho3, m3, P3) * dr
        m_k4 = der_m(r3, rho3) * dr

        phi_k4 = der_phi(der_P(r3, rho3, m3, P3), rho3, P3) * dr
        omega_k4 = der_omega(r3, m3, j3, phi3) * dr
        j_k4 = der_j(r3, rho3, m3, P3, phi3, omega3) * dr
        moi_k4 = der_moi(r3, rho3, m3, P3, phi3, omega3) * dr

        P_f = P + (1 / 6) * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4)
        m_f = m + (1 / 6) * (m_k1 + 2 * m_k2 + 2 * m_k3 + m_k4)

        phi_f = phi + (1 / 6) * (phi_k1 + 2 * phi_k2 + 2 * phi_k3 + phi_k4)
        omega_f = omega + (1 / 6) * (omega_k1 + 2 * omega_k2 + 2 * omega_k3 + omega_k4)
        j_f = j + (1 / 6) * (j_k1 + 2 * j_k2 + 2 * j_k3 + j_k4)
        moi_f = moi + (1 / 6) * (moi_k1 + 2 * moi_k2 + 2 * moi_k3 + moi_k4)

        if P_f < 0:
            P_f = P_min
            rho_f = 10 ** rho_interp(np.log10(P_f))
            break

        rho_f = 10 ** rho_interp(np.log10(P_f))

        H_m = m / der_m(r, rho)
        H_p = P / np.abs(der_P(r, rho, m, P))
        H = (H_m * H_p) / (H_m + H_p)
        dr = max(0.02 * H, 100)  # do not allow dr to be smaller than 100 cm

        r = r3
        m = m_f
        P = P_f
        rho = rho_f
        phi = phi_f
        omega = omega_f
        j = j_f
        moi = moi_f

        compactness = 2 * G * m / (r * c ** 2)
        Ricci_scalar = (k * (rho * c ** 2 - 3 * P))
        Weyl_scalar = ((4 / 3) * (((6 * G * m) / ((c ** 2) * (r ** 3))) - (k * rho * c ** 2)) ** 2) ** 0.5
        fully_contracted_Ricci = ((k ** 2) * (((rho * c ** 2) ** 2) + 3 * P ** 2)) ** 0.5
        Kretschmann_scalar = ((k ** 2) * (3 * (rho * c ** 2) ** 2 + 3 * P ** 2 + 2 * P * rho * c ** 2) - k * (
                16 * G * m / ((r ** 3) * (c ** 2))) * (rho * c ** 2) + (48 *
                                                                        (G ** 2) * m ** 2) / (
                                      (r ** 6) * (c ** 4))) ** 0.5




        i = i + 1

    Mass = m / m_sun
    Radius = r / km
    compactness = 2 * G * m / (r * c ** 2)

    return Mass, Radius, stiffness, compactness, P_c, omega, j, moi, Ricci_scalar, Weyl_scalar, \
           fully_contracted_Ricci, Kretschmann_scalar

    # -----------------------------------------------


if flag == 0:
    Mass, Radius, stiffness, compactness, P_c, omega, j, moment_of_inertia, Ricci_scalar, Weyl_scalar, \
    fully_contracted_Ricci, Kretschmann_scalar = radial(rho_c)
    f.close()
    g.close()
    print('A neutron star with central energy density ', rho_max, ' g/cm^3')
    print('Mass is:', Mass)
    print('Radius is:', Radius)
    print('Compactness is:', compactness)
    print('Stiffness is:', stiffness)
    print('omega is:', omega)
    print('j is:', j)
    print('moment of inertia is:', moment_of_inertia)

if flag == 1:
    rho_central = np.logspace(np.log10(0.1 * rho_max), np.log10(rho_max), num=200, endpoint=True, base=10)
    f = open("MR8.out", 'a')
    count = 0
    for rho_c in rho_central:
        Mass, Radius, stiffness, compactness, P, omega, j, moi, Ricci_scalar, Weyl_scalar, \
        fully_contracted_Ricci, Kretschmann_scalar = radial(rho_c)
        print(count, rho_c / 1E15, P / 1e35, Mass, Radius, moi)
        dat = [rho_c, Mass, Radius, stiffness, compactness, moi]
        np.savetxt(f, np.column_stack(dat), delimiter=" ", fmt='%1.4e')
        count += 1
    f.close()

