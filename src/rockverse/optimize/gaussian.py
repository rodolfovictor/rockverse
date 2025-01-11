import numpy as np
from scipy.integrate import cumtrapz
from scipy.optimize import minimize

__all__ = ['gaussian_val',
           'gaussian_fit',
           'multi_gaussian_val',
           'multi_gaussian_auto_guess',
           'multi_gaussian_fit']
#%%
def gaussian_val(c, x):
    return c[0]*np.exp(-0.5*((x-c[1])/c[2])**2)

def error_gaussian(c, x, y, order=1):
    return np.linalg.norm(y-gaussian_val(c, x), ord=order)

def gaussian_fit(x, y, *, center_bounds=None, order=1):
    if center_bounds is not None:
        lim = [min(center_bounds), max(center_bounds)]
    else:
        lim = None
    ind = np.logical_and(x>=min(lim), x<=max(lim))
    if not any(ind):
        raise Exception('No x value inside the interval defined by center_bounds.')
    xset = x[ind]
    yset = y[ind]
    ysum = cumtrapz(yset, xset, initial=0)
    ysum /= max(ysum)
    p = np.interp([0.025, 0.5, 0.975], ysum, xset)
    c = [max(y), p[1], (p[2]-p[0])/4]
    fun = minimize(error_gaussian, x0=c, args=(x, y, order),
                   bounds=((0, None), lim, ((p[2]-p[0])/100, None)))
    #if not fun.success:
    #    raise Exception('Gaussian fit did not converge.')
    return fun.x

def multi_gaussian_val(c, x):
    y = 0*x
    for k in range(c.shape[0]):
        y += gaussian_val(c[k, :], x)
    return y

def error_multi_gaussian(c, x, y, order=1):
    c0 = np.reshape(c, (-1, 3), order='C')
    return np.linalg.norm(y-multi_gaussian_val(c0, x), ord=order)

def multi_gaussian_auto_guess(x, y, N, order=1):
    c = []
    ya = y.copy()
    for k in range(N):
        x0 = x[np.argmax(ya)]
        c.append(gaussian_fit(x, ya, limits=[0.9*x0, 1.1*x0], order=order))
        ya -= gaussian_val(c[k], x)
    c = np.array(c)
    ind = np.argsort(c[:, 1])
    return c[ind, :]

def multi_gaussian_fit(x, y, c0=None, order=2):
    if isinstance(c0, int):
        c00 = multi_gaussian_auto_guess(x, y, c0, order=order)
    else:
        c00 = c0.copy()
    c = c00.flatten(order='C')
    fun = minimize(error_multi_gaussian, x0=c, args=(x, y, order),
                   bounds=((0, None),)*len(c))
    if not fun.success:
        raise Exception('Gaussian fit did not converge. Try adjusting the initial guess.')
    cfit = np.reshape(fun.x, (-1, 3), order='C')
    ind = np.argsort(cfit[:, 1])
    return cfit[ind, :]
