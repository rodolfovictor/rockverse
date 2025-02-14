import numpy as np
from scipy.optimize import minimize

def gaussian_val(c, x):
    return c[0]*np.exp(-0.5*((x-c[1])/c[2])**2)


def _gaussian_fit_logspace(x, y):

    #get sorted data
    ind = np.argsort(x)
    xs = np.array(x[ind]).astype(float)
    ys = np.array(y[ind]).astype(float)

    # get only positive values and go to logspace
    xp = xs[ys>0]
    yp = np.log(ys[ys>0])

    #normalize
    minxp = min(xp)
    ranxp = max(xp)-min(xp)
    minyp = min(yp)
    ranyp = max(yp)-min(yp)
    xn = (xp-minxp)/ranxp
    yn = (yp-minyp)/ranyp

    # Polynomial fit on normalized data
    m = len(xn)
    A = np.ones((m, 3), dtype=float)
    A[:, 0] = xn*xn
    A[:, 1] = xn
    cn = np.linalg.lstsq(A, yn)[0]

    # Back to true data on log space
    cl = np.array([0, 0, 0], dtype=float)
    cl[0] = ranyp*cn[0]/ranxp/ranxp
    cl[1] = -cn[0]*ranyp*2*minxp/ranxp/ranxp + ranyp*cn[1]/ranxp
    cl[2] = cn[0]*ranyp*minxp*minxp/ranxp/ranxp -ranyp*cn[1]*minxp/ranxp +cn[2]*ranyp +minyp

    # Back to true data in linear space
    c = np.array([0, 0, 0], dtype=float)
    c[2] = np.sqrt(-1/2/cl[0])
    c[1] = cl[1]*c[2]*c[2]
    c[0] = np.exp(cl[2]+0.5*((c[1]/c[2])**2))

    return c


def error_gaussian(c, x, y, order=1):
    return np.linalg.norm(y-gaussian_val(c, x), ord=order)


def gaussian_fit(x, y, order=1):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be numpy arrays.")

    #Data conditioning
    minx = min(x)
    rangex = max(x)-minx
    rangey = max(y)-min(y)
    x_cond = (x-minx)/rangex
    y_cond = y/rangey
    i = np.argmax(y_cond)
    imin = max(i-2, 0)
    imax = min(imin+5, len(y_cond))
    c0 = _gaussian_fit_logspace(x_cond[imin:imax], y_cond[imin:imax])

    fun = minimize(error_gaussian, x0=c0, args=(x_cond, y_cond, order))

    #Back to original data
    c = fun.x
    c[0] *= rangey
    c[1] = minx + c[1]*rangex
    c[2] *= rangex
    return c

def multi_gaussian_val(c, x):
    y = 0*x
    for k in range(c.shape[0]):
        y += gaussian_val(c[k, :], x)
    return y

def error_multi_gaussian(c, x, y, order=1):
    c0 = np.reshape(c, (-1, 3), order='C')
    return np.linalg.norm(y-multi_gaussian_val(c0, x), ord=order)

def multi_gaussian_auto_guess(x, y, n):
    c = []
    ya = y.copy()
    for k in range(n):
        x0 = x[np.argmax(ya)]
        c.append(gaussian_fit(x, ya, center_bounds=[0.9*x0, 1.1*x0], order=1))
        #c.append(gaussian_fit(x, ya, center_bounds=[0.9*x0, 1.1*x0], order=1))
        ya -= gaussian_val(c[k], x)
    c = np.array(c)
    ind = np.argsort(c[:, 1])
    return c[ind, :]

def multi_gaussian_fit(x, y, c0, order=2):
    if isinstance(c0, int):
        c00 = multi_gaussian_auto_guess(x, y, n=c0)
    else:
        c00 = c0.copy()
    c = c00.flatten(order='C')
    fun = minimize(error_multi_gaussian, x0=c, args=(x, y, order),
                   bounds=((0, None),)*len(c))
    cfit = np.reshape(fun.x, (-1, 3), order='C')
    ind = np.argsort(cfit[:, 1])
    return cfit[ind, :]

def multi_gaussian_plot(c, x):
    plt.plot(x, multi_gaussian_val(c, x), 'k')
    for k in range(c.shape[0]):
        plt.plot(x, gaussian_val(c[k, :], x))
