import numpy as np
from numba import cuda, types
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64

from rockverse.dect._corefunctions import calc_rho_Z_gpu
from rockverse.dect._corefunctions import calcABn_gpu


@cuda.jit()
def coeff_matrix_broad_search_gpu(rng_states, matrix, Z1v, Z2v, Z3v, args, cdfx0, cdfy0, cdfx1, cdfy1, cdfx2, cdfy2, cdfx3, cdfy3):

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

    rho1, rho2, rho3, maxA, maxB, maxn, tol = args
    k = cuda.grid(1)
    if k<0 or k>=matrix.shape[0]:
        return
    CT0, CT1, CT2, CT3, Z1, Z2, Z3, A, B, n, err = matrix[k, :]
    matrix[k, -1] = error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v)
    if matrix[k, -1]<tol:
        return

    for _ in range(100):
        if matrix[k, -1] < tol:
            return
        if len(cdfx0) == 2:
            CT0 = xoroshiro128p_normal_float64(rng_states, k)*cdfx0[1] + cdfx0[0]
        else:
            CT0 = draw_pdf(cdfx0, cdfy0, xoroshiro128p_uniform_float64(rng_states, k))

        if len(cdfx1) == 2:
            CT1 = xoroshiro128p_normal_float64(rng_states, k)*cdfx1[1] + cdfx1[0]
        else:
            CT1 = draw_pdf(cdfx1, cdfy1, xoroshiro128p_uniform_float64(rng_states, k))

        if len(cdfx2) == 2:
            CT2 = xoroshiro128p_normal_float64(rng_states, k)*cdfx2[1] + cdfx2[0]
        else:
           CT2 = draw_pdf(cdfx2, cdfy2, xoroshiro128p_uniform_float64(rng_states, k))

        if len(cdfx3) == 2:
            CT3 = xoroshiro128p_normal_float64(rng_states, k)*cdfx3[1] + cdfx3[0]
        else:
            CT3 = draw_pdf(cdfx3, cdfy3, xoroshiro128p_uniform_float64(rng_states, k))

        for _ in range(100):
            A0 = xoroshiro128p_uniform_float64(rng_states, k)*maxA + 1e-10
            B0 = xoroshiro128p_uniform_float64(rng_states, k)*maxB + 1e-10
            n0 = xoroshiro128p_uniform_float64(rng_states, k)*maxn + 1e-10
            A, B, n, err, Z1, Z2, Z3 = calcABn_gpu(CT0, CT1, CT2, CT3,
                                                   rho1, rho2, rho3,
                                                   Z1v, Z2v, Z3v,
                                                   A0, B0, n0, tol)
            if (err<tol and A>0 and B>0 and n>0 and Z1>0 and Z2>0 and Z3>0):
                matrix[k, 0] = CT0
                matrix[k, 1] = CT1
                matrix[k, 2] = CT2
                matrix[k, 3] = CT3
                matrix[k, 4] = Z1
                matrix[k, 5] = Z2
                matrix[k, 6] = Z3
                matrix[k, 7] = A
                matrix[k, 8] = B
                matrix[k, 9] = n
                matrix[k, 10] = err
                return


@cuda.jit()
def reset_arrays_gpu(darray_rho, darray_Z, darray_error):
    nx, = darray_rho.shape
    i = cuda.grid(1)
    if i >= 0 and i < nx:
        darray_rho[i] = 0.
        darray_Z[i] = 0.
        darray_error[i] = 1.



@cuda.jit()
def calc_rhoZ_arrays_gpu(darray_rho, darray_Z, darray_error, dmatrixl, dmatrixh,
                         rng_states, CTl, CTh, rho1, rho2, rho3,
                         required_iterations, tol):
    nx, = darray_rho.shape
    i = cuda.grid(1)
    if not (i >= 0 and i < nx):
        return
    maxiter = dmatrixl.shape[0]
    lim = min(required_iterations, maxiter)
    draw = 0
    while draw < lim:
        #random acces to coefficient matrices
        ind = int(maxiter*xoroshiro128p_uniform_float64(rng_states, i))
        CT0l, CT1l, CT2l, CT3l, Z1l, Z2l, Z3l, Al, Bl, nl, errl = dmatrixl[ind, :]
        CT0h, CT1h, CT2h, CT3h, Z1h, Z2h, Z3h, Ah, Bh, nh, errh = dmatrixh[ind, :]
        rho, Z, err = calc_rho_Z_gpu(rho1, rho2, rho3, CT1h, CT2h, CT3h, CTl, CTh,
                                     CT0l, Al, Bl, nl, CT0h, Ah, Bh, nh, tol)
        if err < tol:
            darray_rho[i] = rho
            darray_Z[i] = Z
            darray_error[i] = err
            break
        draw += 1
    darray_rho[i] = rho
    darray_Z[i] = Z
    darray_error[i] = err
