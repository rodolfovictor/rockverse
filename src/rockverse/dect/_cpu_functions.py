import numpy as np
from numba import njit
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.dect._corefunctions import calcABn_cpu
from rockverse.dect._corefunctions import calc_rho_Z_cpu


@njit()
def fill_coeff_matrix_cpu(matrix, Z1v, Z2v, Z3v, args, cdfx0, cdfy0, cdfx1, cdfy1, cdfx2, cdfy2, cdfx3, cdfy3):

    def draw_pdf(cdfx, cdfy, cdfvalue):
        pdf = None
        #Try linear interpolation
        for k in range(1, len(cdfx)):
            if cdfy[k-1] <= cdfvalue < cdfy[k]:
                m = (cdfx[k]-cdfx[k-1])/(cdfy[k]-cdfy[k-1])
                pdf = cdfx[k] + m*(cdfy[k]-cdfy[k-1])
                break
        if pdf is None: #fall back to closest
            min_dist = abs(cdfy[0]-cdfvalue)
            pdf = cdfx[0]
            for k in range(1, len(cdfx)):
                if abs(cdfy[k]-cdfvalue)<min_dist:
                    min_dist = abs(cdfy[k]-cdfvalue)
                    pdf = cdfx[k]
        return pdf

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

    rho1, rho2, rho3, maxA, maxB, maxn, tol = args
    #Broad random search
    for k in range(matrix.shape[0]):
        CT0, CT1, CT2, CT3, Z1, Z2, Z3, A, B, n, err = matrix[k, :]
        matrix[k, -1] = error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v)
        if matrix[k, -1] < tol:
            continue
        for _ in range(100):
            if matrix[k, -1] < tol:
                break

            if len(cdfx0) == 2:
                CT0 = np.random.randn()*cdfx0[1] + cdfx0[0]
            else:
                CT0 = draw_pdf(cdfx0, cdfy0, np.random.rand())

            if len(cdfx1) == 2:
                CT1 = np.random.randn()*cdfx1[1] + cdfx1[0]
            else:
                CT1 = draw_pdf(cdfx1, cdfy1, np.random.rand())

            if len(cdfx2) == 2:
                CT2 = np.random.randn()*cdfx2[1] + cdfx2[0]
            else:
                CT2 = draw_pdf(cdfx2, cdfy2, np.random.rand())

            if len(cdfx3) == 2:
                CT3 = np.random.randn()*cdfx3[1] + cdfx3[0]
            else:
                CT3 = draw_pdf(cdfx3, cdfy3, np.random.rand())

            for _ in range(100):
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
def calc_rhoZ_arrays_cpu(required_iterations, matrixl, matrixh, CTl, CTh, rho1, rho2, rho3, tol):
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
