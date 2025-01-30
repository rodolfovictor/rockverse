import numpy as np
from scipy.optimize import minimize


#%%
def gaussian_val(c, x):
    return c[0]*np.exp(-0.5*((x-c[1])/c[2])**2)

def error_gaussian(c, x, y, order=1):
    return np.linalg.norm(y-gaussian_val(c, x), ord=order)

def gaussian_fit(x, y, *, center_bounds=None, order=1):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be numpy arrays.")

    #Data conditioning
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    rangex = maxx-minx
    rangey = maxy-miny
    x_cond = (x-minx)/rangex
    y_cond = (y-miny)/rangey
    if center_bounds is not None:
        lim_mu = np.array([min(center_bounds), max(center_bounds)])
        lim_mu -= minx
        lim_mu /= rangex
        mean_guess = np.mean(lim_mu)
    else:
        lim_mu = (None, None)
        mean_guess = np.sum(y_cond*x_cond)/np.sum(y_cond)
    var_guess = np.sqrt(np.sum((y_cond*x_cond-mean_guess)**2))/np.sum(y_cond)

    c_cond = [max(y_cond), mean_guess, var_guess]
    fun_cond = minimize(error_gaussian,
                        x0=c_cond,
                        args=(x_cond, y_cond, order),
                        bounds=((1e-10*c_cond[0], None),
                                lim_mu,
                                (1e-10*c_cond[2], None)),
                        method='Powell')

    cfit = [fun_cond.x[0] * rangey + miny,
            fun_cond.x[1] * rangex + minx,
            fun_cond.x[2] * rangex]
    if lim_mu[0] is not None:
        lim_mu *= rangex
        lim_mu += minx

    fun = minimize(error_gaussian, x0=cfit, args=(x, y, order),
                   bounds=((1e-10*cfit[0], None),
                           lim_mu,
                           (1e-10*cfit[2], None)),
                   method='Powell')
    return fun.x

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
