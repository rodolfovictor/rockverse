import numpy as np
from rockverse._assert.utils import collective_raise

#%% Iterables
# def any_iterable(varname, var):
#     if not hasattr(var, "__iter__"):
#         collective_raise(ValueError(f"Expected iterable for {varname}."))

def any_iterable_non_negative_integers(varname, var):
    if (not hasattr(var, "__iter__")
        or any(np.dtype(type(k)).kind not in 'iu' for k in var)
        or any(k<0 for k in var)):
        collective_raise(ValueError(f"Expected iterable with non negative integers for {varname}."))

# def elements_of_type(varname, var, typenames, types):
#     any_iterable(varname, var)
#     if not all(isinstance(k, types) for k in var):
#         collective_raise(ValueError(f"Expected elements of {typenames} for {varname}."))

def length(varname, var, length):
    if not (hasattr(var, '__iter__') and len(var)==length):
        collective_raise(ValueError(f'Expected {length}-element iterable for {varname}.'))

# def each_in_group(varname, var, group):
#     if any(k not in group for k in var):
#         collective_raise(ValueError(f"{varname} entries must be one of {tuple(group)}."))

# def ordered_iterable(varname, var):
#     if not (hasattr(var, '__iter__') and hasattr(var, '__getitem__')):
#         collective_raise(ValueError(f"Expected ordered iterable for {varname}."))

# def ordered_integers(varname, var):
#     if not (hasattr(var, '__iter__')
#             and hasattr(var, '__getitem__')
#             and all(np.dtype(type(k)).kind in 'ui' for k in var)):
#         collective_raise(ValueError(f'Expected ordered iterable with integer values for {varname}.'))

def ordered_integers_positive(varname, var):
    if not (hasattr(var, '__iter__')
            and hasattr(var, '__getitem__')
            and all(np.dtype(type(k)).kind in 'ui' for k in var)
            and all(k>0 for k in var)):
        collective_raise(ValueError(f'Expected ordered iterable with positive integer values for {varname}.'))

def ordered_numbers(varname, var):
    if not (hasattr(var, '__iter__')
            and hasattr(var, '__getitem__')
            and all(np.dtype(type(k)).kind in 'uif' for k in var)):
        collective_raise(ValueError(f'Expected ordered iterable with numerical values for {varname}.'))

def ordered_numbers_positive(varname, var):
    if not (hasattr(var, '__iter__')
            and hasattr(var, '__getitem__')
            and all(np.dtype(type(k)).kind in 'uif' for k in var)
            and all(k>0 for k in var)):
        collective_raise(ValueError(f'Expected ordered iterable with positive numerical values for {varname}.'))