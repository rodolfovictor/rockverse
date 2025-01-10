import numpy as np
from numba import njit
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

from rockverse.dualenergyct._corefunctions import calcABn_cpu
from rockverse.dualenergyct._corefunctions import calc_rho_Z_cpu



@njit()
def _fill_coeff_matrix_cpu(matrix, Z1v, Z2v, Z3v, args):

    def Zn(values, n):
        sum_ = np.float32(0)
        for k in range(values.shape[0]):
            sum_ += values[k, 2]*values[k, 0]
        v = np.float32(0)
        for k in range(values.shape[0]):
            v += values[k, 2]*values[k, 0]/sum_*(values[k, 0]**n)
        return v

    def error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v):
        F1 = A+B*Zn(Z1v, n)-(CT1-CT0)/rho1
        F2 = A+B*Zn(Z2v, n)-(CT2-CT0)/rho2
        F3 = A+B*Zn(Z3v, n)-(CT3-CT0)/rho3
        err = (F1*F1+F2*F2+F3*F3)**0.5
        return err

    m0, s0, m1, s1, m2, s2, m3, s3, rho1, rho2, rho3, maxA, maxB, maxn, tol = args

    #Broad random search
    for k in range(matrix.shape[0]):
        CT0, CT1, CT2, CT3, Z1, Z2, Z3, A, B, n, err = matrix[k, :]
        matrix[k, -1] = error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v)
        if matrix[k, -1] < tol:
            continue
        for _ in range(1000):
            if matrix[k, -1] < tol:
                break
            CT0 = m0 + s0*np.random.randn()
            CT1 = m1 + s1*np.random.randn()
            CT2 = m2 + s2*np.random.randn()
            CT3 = m3 + s3*np.random.randn()
            for _ in range(1000):
                A0 = np.random.rand()*maxA + 1e-10
                B0 = np.random.rand()*maxB + 1e-10
                n0 = np.random.rand()*maxn + 1e-10
                A, B, n, err, Z1, Z2, Z3 = calcABn_cpu(CT0, CT1, CT2, CT3,
                                                       rho1, rho2, rho3,
                                                       Z1v, Z2v, Z3v,
                                                       A0, B0, n0, tol)
                if (err<tol and A>0 and B>0 and n>0 and Z1>0 and Z2>0 and Z3>0):
                    matrix[k, :] = [CT0, CT1, CT2, CT3, Z1, Z2, Z3, A, B, n, err]
                    break


@njit()
def _calc_rhoZ_arrays_cpu(required_iterations, matrixl, matrixh, CTl, CTh, m0l, s0l, m0h, s0h, rho1, rho2, rho3, tol):
    array_rho = np.nan*np.zeros(matrixl.shape[0], dtype='f8')
    array_Z = np.nan*np.zeros(matrixl.shape[0], dtype='f8')
    array_error = np.nan*np.zeros(matrixl.shape[0], dtype='f8')

    #solve through auto-guess Newton-Raphson for each line
    valid = 0
    for l in range(matrixh.shape[0]):
        CT0l, CT1l, CT2l, CT3l, Z1l, Z2l, Z3l, Al, Bl, nl, errl = matrixl[l, :]
        CT0h, CT1h, CT2h, CT3h, Z1h, Z2h, Z3h, Ah, Bh, nh, errh = matrixh[l, :]

        rho, Z, err = calc_rho_Z_cpu(rho1, rho2, rho3,
                                     CT1h, CT2h, CT3h,
                                     CTl, CTh,
                                     CT0l, Al, Bl, nl,
                                     CT0h, Ah, Bh, nh,
                                     tol)
        array_rho[l] = rho
        array_Z[l] = Z
        array_error[l] = err
        if err < tol:
            valid += 1
            if valid >= required_iterations:
                break
    return array_rho, array_Z, array_error
