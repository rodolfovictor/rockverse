import numpy as np
from numba import njit, cuda

def Zn(values, n):
    sum_ = np.float64(0)
    for k in range(values.shape[0]):
        sum_ += values[k, 2]*values[k, 0]
    v = np.float64(0)
    for k in range(values.shape[0]):
        v += values[k, 2]*values[k, 0]/sum_*(values[k, 0]**n)
    return v

def error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v):
    F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
    F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
    F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
    err = (F1*F1+F2*F2+F3*F3)**0.5
    return err


def calcABn(CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v, A0, B0, n0, tol):

    def Zn(values, n):
        sum_ = np.float64(0)
        for k in range(values.shape[0]):
            sum_ += values[k, 2]*values[k, 0]
        v = np.float64(0)
        for k in range(values.shape[0]):
            v += values[k, 2]*values[k, 0]/sum_*(values[k, 0]**n)
        return v

    #Initial guess
    A = A0
    B = B0
    n = n0
    F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
    F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
    F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
    err = (F1*F1+F2*F2+F3*F3)**0.5

    F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
    F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
    F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
    J11, J21, J31 = 1.0, 1.0, 1.0
    for _ in range(250):
        J12 = Zn(Z1v, n)
        J22 = Zn(Z2v, n)
        J32 = Zn(Z3v, n)
        J13 = B*( Zn(Z1v, 1.01*n) - Zn(Z1v, 0.99*n) )/0.02/n
        J23 = B*( Zn(Z2v, 1.01*n) - Zn(Z2v, 0.99*n) )/0.02/n
        J33 = B*( Zn(Z3v, 1.01*n) - Zn(Z3v, 0.99*n) )/0.02/n

        D = J11*J22*J33 + J12*J23*J31 + J13*J21*J32 -J31*J22*J13 - J32*J23*J11 - J33*J21*J12
        Dx = F1*J22*J33 + J12*J23* F3 + J13* F2*J32 - F3*J22*J13 - J32*J23* F1 - J33* F2*J12
        Dy = J11* F2*J33 +  F1*J23*J31 + J13*J21* F3 -J31* F2*J13 -  F3*J23*J11 - J33*J21* F1
        Dz = J11*J22* F3 + J12* F2*J31 +  F1*J21*J32 -J31*J22* F1 - J32* F2*J11 - F3*J21*J12

        if abs(D)<1e-10:
            break

        dA = Dx/D
        dB = Dy/D
        dn = Dz/D

        if A<dA or B<dB or n<dn:
            break

        A -= dA
        B -= dB
        n -= dn

        F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
        F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
        F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
        err = (F1*F1+F2*F2+F3*F3)**0.5
        if err<tol:
            break

    Z1 = Zn(Z1v, n)**(1.0/n)
    Z2 = Zn(Z2v, n)**(1.0/n)
    Z3 = Zn(Z3v, n)**(1.0/n)
    F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
    F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
    F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
    err = (F1*F1+F2*F2+F3*F3)**0.5
    return A, B, n, err, Z1, Z2, Z3

calcABn_cpu = njit(calcABn)
calcABn_gpu = cuda.jit(calcABn)



@njit()
def _make_index(index, lowECT, highECT, mask, valid, required_iterations,
               m0l, s0l, m0h, s0h, mpi_nprocs):
    count = -1
    nx, ny, nz = valid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                CTl = np.float64(lowECT[i, j, k])
                CTh = np.float64(highECT[i, j, k])

                #test if 95% CI for air or smaller than air (filler)
                if (abs((CTl-m0l)/s0l)<3
                    or abs((CTh-m0h)/s0h)<3
                    or (CTl<m0l-3*s0l)
                    or (CTh<m0h-3*s0h)):
                    continue

                if not mask[i, j, k] and valid[i, j, k] < required_iterations:
                    count = (count + 1) % mpi_nprocs
                    index[i, j, k] = count



def calc_rho_Z(rho1, rho2, rho3,
                 CT1h, CT2h, CT3h,
                 CTl, CTh,
                 CT0l, Al, Bl, nl,
                 CT0h, Ah, Bh, nh,
                 tol):

    if (CTl<CT0l) or (CTh<CT0h):
        return 0.0, 0.0, 0.0

    #Initial guess for rho: least squares approx only Compton high energy
    #Model mu=A rho +C; rho = A'mu+C'
    J00 = CT1h*CT1h + CT2h*CT2h + CT3h*CT3h
    J01 = CT1h + CT2h + CT3h
    J10 = J01
    J11 = 3.0
    F0 = CT1h*rho1 + CT2h*rho2 + CT3h*rho3
    F1 = rho1 + rho2 + rho3
    det = J00*J11-J01*J10
    if abs(det)<1e-10:
        return 0.0, 0.0, 1.0

    A = (1.0/det)*(J11*F0-J01*F1)
    C = (1.0/det)*(J00*F1-J10*F0)
    rho = A*CTh+C
    if not (rho>0.0):
        return 0.0, 0.0, 1.0

    #Initial guess for Z: sweep based on initial guess for rho
    bestZ = 0
    besterr = np.inf
    Z = 1
    while Z < 120:
        F0 = CTl - rho*( Al + Bl*(Z**nl) ) - CT0l
        F1 = CTh - rho*( Ah + Bh*(Z**nh) ) - CT0h
        err = (F0*F0+F1*F1)**0.5
        if err < besterr:
            besterr = err
            bestZ = Z
        Z += 0.1
    Z = bestZ

    # Newton-Raphson refinment
    F0 = CTl - rho*( Al + Bl*(Z**nl) ) - CT0l
    F1 = CTh - rho*( Ah + Bh*(Z**nh) ) - CT0h
    errF = (F0*F0+F1*F1)**0.5
    for _ in range(1000):
        rho_old = rho
        Z_old = Z
        J00 = -Al -Bl*(Z**nl)
        J10 = -Ah -Bh*(Z**nh)
        J01 = -rho* Bl* nl*(Z**(nl-1))
        J11 = -rho*Bh*nh*(Z**(nh-1))
        det = J00*J11-J01*J10
        if abs(det)<1e-10:
            break
        rho -= (1.0/det)*(J11*F0-J01*F1)
        Z -= (1.0/det)*(J00*F1-J10*F0)
        if not (rho>0 and  Z>0):
            rho, Z = rho_old, Z_old
            break
        F0 = CTl -rho*( Al +Bl*(Z**nl) ) - CT0l
        F1 = CTh -rho*( Ah +Bh*(Z**nh) ) - CT0h
        errF = (F0*F0+F1*F1)**0.5
        errRhoZ = ( (((rho-rho_old)*(rho-rho_old)+(Z-Z_old)*(Z-Z_old))**0.5)
                   /((rho_old*rho_old+Z_old*Z_old)**0.5) )
        if errF<tol or errRhoZ<tol:
            break
    F0 = CTl - rho*(Al + Bl*(Z**nl)) - CT0l
    F1 = CTh - rho*(Ah + Bh*(Z**nh)) - CT0h
    errF = (F0*F0+F1*F1)**0.5

    if not (rho>0 and Z>0):
        return 0.0, 0.0, 1.0

    return rho, Z, errF


calc_rho_Z_cpu = njit(calc_rho_Z)
calc_rho_Z_gpu = cuda.jit(calc_rho_Z)