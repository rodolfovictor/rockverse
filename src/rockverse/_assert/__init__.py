import zarr
from rockverse._assert.utils import collective_raise
import rockverse._assert.condition
import rockverse._assert.iterable

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

def drpdtype(varname, var):
    if var not in ('|b1', '|u1', '|i1',
                   '<u2', '<u4', '<u8', '<u16',
                   '>u2', '>u4', '>u8', '>u16',
                   '<i2', '<i4', '<i8', '<i16',
                   '>i2', '>i4', '>i8', '>i16',
                   '<f4', '<f8', '>f4', '>f8'):
        collective_raise(TypeError(
            f"{varname} must be passed as Numpy typestring, "
            "composed as follows:"
            "\n    - The character for byte order:"
            "\n        - '<' if little endian"
            "\n        - '>' if big endian"
            "\n        - '|' if not applicable (8-bit numbers)"
            "\n    - The character for data type:"
            "\n        - 'i' for signed integer"
            "\n        - 'u' for unsigned integer"
            "\n        - 'f' for floating-point"
            "\n    - The number of bytes per value"
            "\n        - '1' for 8-bits"
            "\n        - '2' for 16-bits"
            "\n        - '4' for 32-bits"
            "\n        - '8' for 64-bits"
            "\n        - '16' for 128-bits"
            "\nExamples:"
            "\n    - dtype='|b1': boolean"
            "\n    - dtype='|u1': 8-bit unsigned integer"
            "\n    - dtype='<i2': 16-bit signed integer, little endian"
            "\n    - dtype='>f4': 32-bit floating point, big endian"
            ))

def rockverse_instance(var, var_name, var_types):
    '''
    varname : str
        Printable variable name
    var : Any
        The variable itself
    vartypes : tuple of strings
        Printable names for the expected variables
    '''
    if (not hasattr(var, '_rockverse_datatype')
        or var._rockverse_datatype not in var_types):
        if len(var_types) == 1:
            collective_raise(TypeError(f"Expected {var_types[0]} for {var_name}."))
        elif len(var_types) == 2:
            collective_raise(TypeError(f"Expected {var_types[0]} or {var_types[1]} for {var_name}."))
        else:
            collective_raise(TypeError(f"Expected {', '.join(var_types[:-1])} or {var_types[-1]} for {var_name}."))

def boolean(varname, var):
    if not isinstance(var, bool):
        collective_raise(ValueError(f"Expected boolean for {varname}."))

def dictionary(varname, var):
    if not isinstance(var, dict):
        collective_raise(ValueError(f"Expected dict for {varname}."))

def instance(varname, var, vartypenames, vartypes):
    if not isinstance(var, vartypes):
        collective_raise(ValueError(f"Expected {vartypenames} for {varname}."))

def dtype(varname, var, vartypenames, vartypes):
   if var.dtype.kind not in vartypes:
       collective_raise(TypeError(f"Expected {vartypenames} dtype for {varname}."))

def in_group(varname, var, group):
    if var in group:
        return
    if len(group) == 1:
        collective_raise(ValueError(f"{varname} must be {group[0]}."))
    elif len(group) == 2:
        collective_raise(ValueError(f"{varname} must be {group[0]} or {group[1]}"))
    else:
        collective_raise(ValueError(f"{varname} must be {', '.join(group[:-1])} or {group[-1]}."))

def list_of_zarray(varname, var):
    conditions = [isinstance(var, list),
                  all(isinstance(k, zarr.core.Array) for k in var)]
    if not all(conditions):
        collective_raise(ValueError(f"Expected list of Zarr arrays for {varname}."))

def same_chunk_size(message, varlist):
   chunks = [k.chunks for k in varlist]
   if not all(ch==chunks[0] for ch in chunks):
       collective_raise(ValueError(f'{message} must have same chunk size.'))

def same_shape(message, varlist):
   shapes = [k.shape for k in varlist]
   if not all(sh==shapes[0] for sh in shapes):
       collective_raise(ValueError(f'{message} must have same shape.'))

def same_shape_if_not_None(message, varlist):
    shapes = [k.shape for k in varlist if k is not None]
    if not all(sh==shapes[0] for sh in shapes):
        collective_raise(ValueError(f'{message} must have same shape.'))

def same_voxel_length(message, varlist):
   lengths1 = varlist[0].voxel_length
   for var in varlist:
       lengths2 = var.voxel_length
       for a, b in zip(lengths1, lengths2):
           m = max(abs(a), abs(b))
           tol = 1e-15*m if m>0 else 1e-15
           if (m>0 and abs(a-b)/m > tol) or (m==0 and abs(a-b) > tol):
               collective_raise(ValueError(f'{message} must have same voxel length.'))

def same_voxel_origin(message, varlist):
   origins1 = varlist[0].voxel_origin
   for var in varlist:
       origins2 = var.voxel_origin
       for a, b in zip(origins1, origins2):
           m = max(abs(a), abs(b))
           tol = 1e-15*m if m>0 else 1e-15
           if (m>0 and abs(a-b)/m > tol) or (m==0 and abs(a-b) > tol):
               collective_raise(ValueError(f'{message} must have same voxel origin.'))

def same_voxel_unit(message, varlist):
   unit = varlist[0].voxel_unit
   for var in varlist:
       if var.voxel_unit != unit:
           collective_raise(ValueError(f'{message} must have same voxel unit.'))

def zarr_array(varname, var):
    if not isinstance(var, zarr.core.Array):
        collective_raise(ValueError(f'Expected Zarr array for {varname}'))
    for n in ('voxel_origin', 'voxel_length', 'voxel_unit'):
        if n not in var.attrs:
            collective_raise(KeyError(f"'{n}' not found in zarr array attributes"))

def zarr_directorystore(varname, var):
    if mpi_nprocs > 1 and not isinstance(var, (str, zarr.storage.DirectoryStore)):
        collective_raise(ValueError(f"Invalid {varname}. Zarr store must be a directory."))

def zarr_or_none_iterable(varname, var):
    if not (hasattr(var, '__iter__')
            and all(k is None or isinstance(k, zarr.core.Array) for k in var)
            ):
        collective_raise(ValueError(f'Expected list of None or Zarr arrays for {varname}.'))