import numpy as np
from numba import cuda, types
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64

from rockverse.dualenergyct._corefunctions import calc_rho_Z_gpu
from rockverse.dualenergyct._corefunctions import calcABn_gpu


@cuda.jit()
def _coeff_matrix_broad_search_gpu(rng_states, matrix, Z1v, Z2v, Z3v, args):

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

    m0, s0, m1, s1, m2, s2, m3, s3, rho1, rho2, rho3, maxA, maxB, maxn, tol = args
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
        CT0 = m0 + s0*xoroshiro128p_normal_float64(rng_states, k)
        CT1 = m1 + s1*xoroshiro128p_normal_float64(rng_states, k)
        CT2 = m2 + s2*xoroshiro128p_normal_float64(rng_states, k)
        CT3 = m3 + s3*xoroshiro128p_normal_float64(rng_states, k)
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

def _fill_coeff_matrix_gpu(matrix, Z1v, Z2v, Z3v, args):
     threadsperblock = 32
     blockspergrid = int(np.ceil(matrix.shape[0]/threadsperblock))
     rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=1)
     d_matrix  = cuda.to_device(matrix)
     d_Z1v = cuda.to_device(Z1v)
     d_Z2v = cuda.to_device(Z2v)
     d_Z3v = cuda.to_device(Z3v)
     d_args = cuda.to_device(args)
     _coeff_matrix_broad_search_gpu[blockspergrid, threadsperblock](rng_states, d_matrix, d_Z1v, d_Z2v, d_Z3v, d_args)
     matrix[:] = d_matrix.copy_to_host()




@cuda.jit()
def _reset_arrays_gpu(darray_rho, darray_Z, darray_error):
    nx, = darray_rho.shape
    i = cuda.grid(1)
    if i >= 0 and i < nx:
        darray_rho[i] = 0.
        darray_Z[i] = 0.
        darray_error[i] = 1.



@cuda.jit()
def _calc_rhoZ_arrays_gpu(darray_rho, darray_Z, darray_error, dmatrixl, dmatrixh,
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
