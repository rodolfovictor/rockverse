import zarr
from zarr.errors import ContainsArrayError
import numpy as np

from rockverse import _assert
from rockverse.digitalrock.voxel_image._math import _array_math
from rockverse.digitalrock.voxel_image._finneypack import _fill_finney_pack
from rockverse.digitalrock.voxel_image.voxel_image import VoxelImage
#from rockverse.digitalrock.voxel_image.histogram import Histogram
from rockverse._utils import rvtqdm
from rockverse.config import config


from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()


def create(shape,
           dtype,
           chunks=True,
           store=None,
           overwrite=False,
           field_name='',
           field_unit='',
           description='',
           voxel_origin=None,
           voxel_length=None,
           voxel_unit='',
           **kwargs):
    """
    .. _rockverse_digitalrock_create_function:

    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`

    Create empty voxel image.

    Parameters
    ----------
    shape : tuple
        Desired image shape.
    dtype : string or dtype
        NumPy dtype.
    chunks : int or tuple of ints, optional
        Chunk shape. If True, will be guessed from shape and number of processes.
        If False, will be set to shape, i.e., single chunk for the whole array.
        Default is True.
    store : Any, optional
        Any valid `Zarr store <https://zarr.readthedocs.io/en/stable/tutorial.html#storage-alternatives>`_.
        Must be a directory when running with multiple MPI processes.
    overwrite : bool, optional
        If True, delete all pre-existing data in the store at the specified path
        before creating the new image. Default value is False.
    field_name : str, optional
        Name for the stored image or scalar field.
    field_unit : str, optional
        Unit for the stored image or scalar field.
    description : str, optional
        Description for the stored image or scalar field.
    voxel_origin : list or tuple, optional
        Image voxel coordinate origin in units of `voxel_unit`. This is the
        spatial coordinate for the voxel with lowest (x, y, z) coordinates.
        If ``None``, defaults to a tuple of zeros for each image dimension.
    voxel_length : list or tuple, optional
        Image voxel length in each dimension. If ``None``, defaults to a tuple
        of ones for each image dimension.
    voxel_unit : str, optional
        Image voxel length unit.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        `Zarr.creation.create <https://zarr.readthedocs.io/en/stable/api/creation.html#zarr.creation.create>`_ function.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    _assert.iterable.ordered_integers_positive('shape', shape)
    if voxel_length is not None:
        _voxel_length = voxel_length
    else:
        _voxel_length = [1 for k in shape]
    _assert.iterable.ordered_numbers_positive('voxel_length', _voxel_length)
    if voxel_origin is not None:
        _voxel_origin = voxel_origin
    else:
        _voxel_origin = [0 for k in shape]
    _assert.iterable.ordered_numbers('voxel_origin', _voxel_origin)
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))
    _assert.instance('voxel_unit', voxel_unit, 'string', (str,))

    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['chunks'] = chunks
    kwargs['store'] = store
    kwargs['overwrite'] = overwrite
    kwargs['store'] = store
    kwargs['order'] = 'C'

    #Only rank 0 writes metadata to disk
    if isinstance(store, (str, zarr.storage.DirectoryStore)):
        msg = ''
        if mpi_rank == 0:
            try:
                z = zarr.create(**kwargs)
                z.attrs['_ROCKVERSE_DATATYPE'] = 'VoxelImage'
                z.attrs['description'] = description
                z.attrs['field_name'] = field_name
                z.attrs['field_unit'] = field_unit
                z.attrs['voxel_unit'] = voxel_unit
                z.attrs['voxel_origin'] = _voxel_origin
                z.attrs['voxel_length'] = _voxel_length
            except ContainsArrayError as e:
                msg = e.__str__()
                pass
        msg = comm.bcast(msg, root=0)
        if msg:
            if mpi_rank == 0:
                raise ContainsArrayError(msg)
            exit(1)
        comm.barrier()
        z = zarr.open(store, 'r+')
    else:
        z = zarr.create(**kwargs)
        z.attrs['_ROCKVERSE_DATATYPE'] = 'VoxelImage'
        z.attrs['description'] = description
        z.attrs['field_name'] = field_name
        z.attrs['field_unit'] = field_unit
        z.attrs['voxel_unit'] = voxel_unit
        z.attrs['voxel_origin'] = _voxel_origin
        z.attrs['voxel_length'] = _voxel_length
    comm.barrier()
    return VoxelImage(store=z.store)


def empty(shape, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create an empty voxel image with the given shape.
    The array will be initialized without any specific values.

    Parameters
    ----------
    shape : tuple
        Desired image shape.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['fill_value'] = None
    return create(**kwargs)


def zeros(shape, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create a voxel image filled with zeros.

    Parameters
    ----------
    shape : tuple
        Desired image shape.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """

    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['fill_value'] = 0
    return create(**kwargs)

def ones(shape, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create a voxel image filled with ones.

    Parameters
    ----------
    shape : tuple
        Desired image shape.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['fill_value'] = 1
    return create(**kwargs)

def full(shape, dtype, fill_value, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create a voxel image filled with a specified value.

    Parameters
    ----------
    shape : tuple
        Desired image shape.
    dtype : string or dtype
        NumPy dtype.
    fill_value : scalar
        Value to fill the array.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """

    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['fill_value'] = fill_value
    return create(**kwargs)


def _put_meta(a, kwargs):
    if 'voxel_origin' not in kwargs:
        kwargs['voxel_origin'] = a.attrs['voxel_origin']
    if 'voxel_unit' not in kwargs:
        kwargs['voxel_unit'] = a.attrs['voxel_unit']
    if 'voxel_length' not in kwargs:
        kwargs['voxel_length'] = a.attrs['voxel_length']


def empty_like(a, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create empty voxel image with same shape, chunks, voxel_origin,
    voxel_length, and voxel_unit as the given image.

    Parameters
    ----------
    a : VoxelImage
        The source voxel image to mimic.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    _assert.rockverse_instance(a, 'a', ('VoxelImage',))
    _put_meta(a, kwargs)
    kwargs['dtype'] = dtype
    return empty(**kwargs)

def zeros_like(a, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create voxel image filled with zeros with same shape, chunks, voxel_origin,
    voxel_length, and voxel_unit as the given image.


    Parameters
    ----------
    a : VoxelImage
        The source RockVerse voxel image to mimic.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    _assert.rockverse_instance(a, 'a', ('VoxelImage',))
    _put_meta(a, kwargs)
    kwargs['dtype'] = dtype
    return zeros(**kwargs)

def ones_like(a, dtype, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create voxel image filled with ones with same shape, chunks, voxel_origin,
    voxel_length, and voxel_unit as the given image.

    Parameters
    ----------
    a : VoxelImage
        The source RockVerse voxel image to mimic.
    dtype : string or dtype
        NumPy dtype.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    _assert.rockverse_instance(a, 'a', ('VoxelImage',))
    _put_meta(a, kwargs)
    kwargs['dtype'] = dtype
    return ones(**kwargs)

def full_like(a, dtype, fill_value, **kwargs):
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`
    Create voxel image filled with a specified value with same shape, chunks,
    voxel_origin, voxel_length, and voxel_unit as the given image.

    Parameters
    ----------
    a : VoxelImage
        The source RockVerse voxel image to mimic.
    dtype : string or dtype
        NumPy dtype.
    fill_value : scalar
        Value to fill the array.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """
    _assert.rockverse_instance(a, 'a', ('VoxelImage',))
    _put_meta(a, kwargs)
    kwargs['fill_value'] = fill_value
    kwargs['dtype'] = dtype
    return full(**kwargs)


def sphere_pack(shape,
                dtype='u1',
                xlim=(-10, 10),
                ylim=(-10, 10),
                zlim=(-10, 10),
                sphere_radius=1,
                fill_value=1,
                **kwargs):
    '''
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`

    Create a sphere pack image.

    John Finney's experimental disordered packing of equal, hard spheres,
    optically imaged in 1970, is hard-coded in RockVerse from the sphere
    centers available in
    `Digital Rocks Portal <https://www.digitalrocksportal.org/projects/47>`_.
    The volumetric image is analytically built in normalized coordinates, with
    sphere radius equal to 1 and (x, y, z) positions for sphere centers roughly
    between -20 and 20 in each direction. The image can be built at any needed
    resolution by a combination of shape and spatial limits.

    .. note::
        If you use this data, please remember to
        `cite the data source <https://www.digitalrocksportal.org/projects/47/cite/>`_
        and the
        `related publications <https://www.digitalrocksportal.org/projects/47/publications/>`_.

    Parameters
    ----------
    shape : tuple
        Shape of the image to create.
    dtype : string or dtype, optional
        NumPy dtype. Default 8-bit unsigned integer.
    xlim : tuple, optional
        Spatial limits of the sphere pack in the x-direction.
    ylim : tuple, optional
        Spatial limits of the sphere pack in the y-direction.
    zlim : tuple, optional
        Spatial limits of the sphere pack in the z-direction.
    sphere_radius : float, optional
        Radius of the spheres to pack. Default is 1 (original pack, touching,
        non-overlapping hard spheres). Increase this value to simulate
        overlapping spheres (cement growth, for example) or decrease to use
        smaller spheres (like grain dissolution). Sphere centers will not
        change.
    fill_value : scalar, optional
        Value to fill the spheres with. Default is 1.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    '''
    ox, fx = xlim
    oy, fy = ylim
    oz, fz = zlim
    nx, ny, nz = shape
    xm = np.linspace(ox, fx, nx)
    hx = xm[1]-xm[0]
    ym = np.linspace(oy, fy, ny)
    hy = ym[1]-ym[0]
    zm = np.linspace(oz, fz, nz)
    hz = zm[1]-zm[0]
    kwargs['voxel_origin'] = [ox, oy, oz]
    kwargs['voxel_length'] = [hx, hy, hz]
    z = zeros(shape, dtype, **kwargs)
    _fill_finney_pack(array=z, sphere_radius=sphere_radius,
                      hx=hx, hy=hy, hz=hz, ox=ox, oy=oy, oz=oz,
                      fill_value=fill_value)
    return z


def import_raw(rawfile,
               shape,
               dtype,
               offset=0,
               raw_file_order='F',
               description='',
               field_name='',
               field_unit='',
               voxel_origin=None,
               voxel_length=None,
               voxel_unit='',
               overwrite=False,
               **kwargs):

    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`

    Import raw file as a voxel image.

    This function imports a raw binary file into a voxel image, adding the
    necessary metadata specific to digital rock petrophysics in RockVerse.
    It supports parallel reading to efficiently import large datasets.

    Parameters
    ----------
    rawfile : str
        Path to the raw file in the file system.
    shape : tuple, list, or numpy.ndarray of ints
        Shape of the array to be created.
    dtype : str
        Data type of the raw file, specified as a Numpy typestring formed by
        the sequence of characters:

        1. The character for byte order:
            * ``<`` if little endian
            * ``>`` if big endian
            * ``|`` if not applicable (8-bit numbers)
        2. The character for data type:
            * ``b`` for boolean
            * ``i`` for signed integer
            * ``u`` for unsigned integer
            * ``f`` for floating-point
        3. The number of bytes per voxel
            * ``1`` for 8-bit data
            * ``2`` for 16-bit data
            * ``4`` for 32-bit data
            * ``8`` for 64-bit data

        These are the accepted combinations:

        * ``'|b1'``: boolean
        * ``'|u1'``: 8-bit unsigned integer
        * ``'|i1'``: 8-bit signed integer

        * ``'<u2'``: little endian 16-bit unsigned integer
        * ``'<u4'``: little endian 32-bit unsigned integer
        * ``'<u8'``: little endian 64-bit unsigned integer
        * ``'<u16'``: little endian 128-bit unsigned integer

        * ``'>u2'``: big endian 16-bit unsigned integer
        * ``'>u4'``: big endian 32-bit unsigned integer
        * ``'>u8'``: big endian 64-bit unsigned integer
        * ``'>u16'``: big endian 128-bit unsigned integer

        * ``'<i2'``: little endian 16-bit signed integer
        * ``'<i4'``: little endian 32 -bit signed integer
        * ``'<i8'``: little endian 64-bit signed integer
        * ``'<i16'``: little endian 128-bit signed integer

        * ``'>i2'``: big endian 16-bit signed integer
        * ``'>i4'``: big endian 32-bit signed integer
        * ``'>i8'``: big endian 64-bit signed integer
        * ``'>i16'``: big endian 128-bit signed integer

        * ``'<f4'``: little endian 32-bit float
        * ``'<f8'``: little endian 64-bit float

        * ``'>f4'``: big endian 32-bit float
        * ``'>f8'``: big endian 64-bit float

        The imported data will be converted to the system's native byte order
        (little endian or big endian).
    offset : int, optional
        Number of bytes to skip at the beginning of the file. Typically a
        multiple of the byte-size of dtype. Default is 0.
    raw_file_order : {'C', 'F'}, optional
        Memory layout order of the raw file: 'C' for C-style (row-major), 'F'
        for Fortran-style (column-major). Use 'F' when loading raw files
        exported from Fiji/ImageJ. After importing, C-style memory layout is
        enforced within chunks for optimal cache performance in Python. Default
        is 'F'.
    description : str, optional
        Description for the stored scalar field.
    field_name : str, optional
        Name for the stored scalar field.
    field_unit : str, optional
        Unit for the stored scalar field.
    voxel_origin : tuple, list, or Numpy array of ints, optional
        Spatial coordinate origin for the first voxel in the array, in units
        of `voxel_unit`. If None, defaults to a tuple of zeros for each array
        dimension.
    voxel_length : tuple, list, or Numpy array of ints or floats, optional
        Voxel length in each direction. If None, defaults to a tuple of ones
        for each array dimension.
    voxel_unit : str, optional
        Unit for the voxel length.
    overwrite : bool, optional
        If True, delete all pre-existing data in the store at the specified
        path before creating the new array. Default is False, to prevent
        accidental overwriting of existing data.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created ``VoxelImage`` object.
    """

    _assert.instance('rawfile', rawfile, 'string', (str,))
    _assert.iterable.ordered_integers_positive('shape', shape)
    if voxel_length is not None:
        _voxel_length = voxel_length
    else:
        _voxel_length = [1 for k in shape]
    _assert.iterable.ordered_numbers_positive('voxel_length', _voxel_length)
    if voxel_origin is not None:
        _voxel_origin = voxel_origin
    else:
        _voxel_origin = [0 for k in shape]
    _assert.iterable.ordered_numbers('voxel_origin', _voxel_origin)
    _assert.instance('dtype', dtype, 'string', (str,))
    _assert.drpdtype('dtype', dtype)
    _assert.instance('name', field_name, 'string', (str,))
    _assert.instance('unit', field_unit, 'string', (str,))
    _assert.instance('offset', offset, 'integer', (int,))
    _assert.in_group('raw_file_order', raw_file_order, ('C', 'F'))
    _assert.instance('voxel_unit', voxel_unit, 'string', (str,))
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))

    kwargs['dtype'] = np.dtype(dtype).kind + str(np.dtype(dtype).itemsize)
    kwargs['description'] = description
    kwargs['field_name'] = field_name
    kwargs['field_unit'] = field_unit
    kwargs['shape'] = shape
    kwargs['voxel_origin'] = _voxel_origin
    kwargs['voxel_length'] = _voxel_length
    kwargs['voxel_unit'] = voxel_unit
    kwargs['overwrite'] = overwrite
    z = create(**kwargs)

    data = np.memmap(rawfile, dtype=dtype, mode='r', offset=offset,
                     shape=tuple(shape), order=raw_file_order)

    desc = 'Importing raw file'
    if kwargs['field_name']:
        desc = f"({kwargs['field_name']}) {desc}"
    for block_id in rvtqdm(range(z.nchunks), desc=desc, unit='chunk'):
        if block_id % mpi_nprocs == mpi_rank:
            box, bex, boy, bey, boz, bez = z.chunk_slice_indices(block_id)
            z[box:bex, boy:bey, boz:bez] = data[box:bex, boy:bey, boz:bez]
    comm.barrier()
    return z
