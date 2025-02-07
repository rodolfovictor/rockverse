import numpy as np
from rockverse.errors import collective_raise

def integer_or_float(varname, var):
    if np.dtype(type(var)).kind in 'fui': #splitting to handle >= op only if number
        return
    collective_raise(ValueError(f"Expected number for {varname}."))

def non_negative_integer_or_float(varname, var):
    if np.dtype(type(var)).kind in 'fui': #splitting to handle >= op only if number
        if var>=0:
            return
    collective_raise(ValueError(f"Expected non negative number for {varname}."))

def positive_integer_or_float(varname, var):
    if np.dtype(type(var)).kind in 'fui': #splitting to handle >= op only if number
        if var>0:
            return
    collective_raise(ValueError(f"Expected positive number for {varname}."))

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

def voxelimage_dtype(varname, var):
    if np.dtype(var).kind not in 'biufc':
        collective_raise(ValueError(
            f"Invalid dtype kind for {varname}. Expected boolean, integer, "
            "unsigned integer, floating-point or complex floating-point."))