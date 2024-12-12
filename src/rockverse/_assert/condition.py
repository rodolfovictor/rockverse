import numpy as np
from rockverse._assert.utils import collective_raise

def integer_or_float(varname, var):
    if np.dtype(type(var)).kind in 'fui': #splitting to handle >= op only if number
        return
    collective_raise(ValueError(f"Expected number for {varname}."))

def non_negative_integer_or_float(varname, var):
    if np.dtype(type(var)).kind in 'fui': #splitting to handle >= op only if number
        if var>=0:
            return
    collective_raise(ValueError(f"Expected non negative number for {varname}."))

def integer(varname, var):
    if np.dtype(type(var)).kind in 'ui':
        return
    collective_raise(ValueError(f"Expected integer for {varname}."))

def non_negative_integer(varname, var):
    if np.dtype(type(var)).kind in 'ui': #splitting to handle >= op only if number
        if var>=0:
            return
    collective_raise(ValueError(f"Expected non negative integer for {varname}."))

def positive_integer(varname, var):
    if np.dtype(type(var)).kind in 'ui': #splitting to handle >= op only if number
        if var>0:
            return
    collective_raise(ValueError(f"Expected positive integer value for {varname}."))