"""
Overview
--------

This module defines the basic class for RockVerse Digital Rock Petrophysics,
the VoxelImage class, intended to contain voxelized images and scalar fields in
general. The VoxelImage class builds upon
`Zarr arrays <https://zarr.readthedocs.io/en/stable/_autoapi/zarr.core.Array.html#zarr.core.Array>`_
by adding attributes and methods specifically designed for digital rock
petrophysics in a high-performance parallel computing environment.

Since it is derived from Zarr arrays, it can efficiently handle large images by
leveraging Zarr's chunked storage. Methods in this class also take advantage of
Zarr's chunked storage for MPI-based parallel processing, optimized for both CPUs
and GPUs using `Numba <https://numba.pydata.org/>`_ 'just-in-time' compilation,
providing flexibility and high-performance computing (HPC) on cluster computers.

The VoxelImage class is designed for simplicity, handling complex computational
abstractions under the hood. This makes it accessible and user-friendly for non-HPC
specialists through high-level functions.

"""

import json
import zarr
from zarr.errors import ContainsArrayError
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

from rockverse import _assert
from rockverse.digitalrock.voxelimage._array_math import _array_math
from rockverse.digitalrock.voxelimage._finneypack import _fill_finney_pack
from rockverse._utils import rvtqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()



class VoxelImage(zarr.Array):

    """
    The basic type for RockVerse Digital Rock Petrophysics, intended to contain
    voxelized images and scalar fields in general. The class builds upon
    `Zarr arrays <https://zarr.readthedocs.io/en/stable/_autoapi/zarr.core.Array.html#zarr.core.Array>`_
    by adding attributes and methods specifically designed for digital rock
    petrophysics in a high performance parallel computing environment.

    .. note::
        This class should not be instantiated directly. Instead, use the provided
        :ref:`creation functions <voxel image creation functions>`.
    """

    @property
    def description(self):
        '''Image or scalar field description.'''
        return self.attrs['description']

    @description.setter
    def description(self, v):
        _assert.instance('v', v, 'str', (str,))
        self.attrs['description'] = v

    @property
    def field_name(self):
        '''Name for the stored scalar field.'''
        return self.attrs['field_name']

    @field_name.setter
    def field_name(self, v):
        _assert.instance('v', v, 'str', (str,))
        self.attrs['field_name'] = v

    @property
    def field_unit(self):
        '''Unit for the stored scalar field.'''
        return self.attrs['field_unit']

    @field_unit.setter
    def field_unit(self, v):
        _assert.instance('v', v, 'str', (str,))
        self.attrs['field_unit'] = v

    @property
    def shape(self):
        return self[...].shape

    @property
    def nx(self):
        '''The number of voxels in x-direction (first axis). Equivalent to shape[0].'''
        return self.shape[0]

    @property
    def ny(self):
        '''The number of voxels in y-direction (second axis). Equivalent to shape[1].'''
        return self.shape[1]

    @property
    def nz(self):
        '''The number of voxels in z-direction (third axis). Equivalent to shape[2].'''
        return self.shape[2]

    @property
    def hx(self):
        '''Voxel length in x-direction (first axis).'''
        return self.attrs['voxel_length'][0]

    @hx.setter
    def hx(self, v):
        _assert.condition.non_negative_integer_or_float('hx', v)
        self.attrs['voxel_length'][0] = v

    @property
    def hy(self):
        '''Voxel length in y-direction (second axis).'''
        return self.attrs['voxel_length'][1]

    @hy.setter
    def hy(self, v):
        _assert.condition.non_negative_integer_or_float('hy', v)
        self.attrs['voxel_length'][1] = v

    @property
    def hz(self):
        '''Voxel length in z-direction (third axis).'''
        return self.attrs['voxel_length'][2]

    @hz.setter
    def hz(self, v):
        _assert.condition.non_negative_integer_or_float('hz', v)
        self.attrs['voxel_length'][2] = v

    @property
    def voxel_length(self):
        """
        Returns the image voxel length in each spatial direction (x, y, z)
        as a tuple. This is a read-only property. To alter the voxel lengths,
        use the ``hx``, ``hy``, and ``hz`` properties.
        """
        return tuple(self.attrs['voxel_length'])

    @property
    def h_unit(self):
        '''Image voxel unit.'''
        return self.attrs['voxel_unit']

    @h_unit.setter
    def h_unit(self, v):
        _assert.instance('h_unit', v, 'string', (str,))
        self.attrs['voxel_unit'] = v

    @property
    def voxel_unit(self):
        '''Image voxel unit.'''
        return self.attrs['voxel_unit']

    @voxel_unit.setter
    def voxel_unit(self, v):
        _assert.instance('voxel_unit', v, 'string', (str,))
        self.attrs['voxel_unit'] = v

    @property
    def dimensions(self):
        '''
        Image total dimension in each direction.

        Returns the total physical dimensions of the image by multiplying
        the number of voxels by the voxel length in each spatial direction (x, y, z).
        '''
        return self.nx*self.hx, self.ny*self.hy, self.nz*self.hz

    @property
    def ox(self):
        '''Spatial x-coordinate for the first voxel in x-direction (first axis).'''
        return self.attrs['voxel_origin'][0]

    @ox.setter
    def ox(self, v):
        _assert.condition.integer_or_float('ox', v)
        self.attrs['voxel_origin'][0] = v

    @property
    def oy(self):
        '''Spatial y-coordinate for the first voxel in y-direction (second axis).'''
        return self.attrs['voxel_origin'][1]

    @oy.setter
    def oy(self, v):
        _assert.condition.integer_or_float('oy', v)
        self.attrs['voxel_origin'][1] = v

    @property
    def oz(self):
        '''Spatial z-coordinate for the first voxel in z-direction (third axis).'''
        return self.attrs['voxel_origin'][2]

    @oz.setter
    def oz(self, v):
        _assert.condition.integer_or_float('oz', v)
        self.attrs['voxel_origin'][2] = v

    @property
    def voxel_origin(self):
        '''
        Image voxel origin for each direction.

        Returns the spatial coordinate origin for the first voxel in the array
        in each spatial direction (x, y, z) as a tuple. This is a read-only
        property. To alter the voxel origins, use the ``ox``, ``oy``, and ``oz``
        properties.
        '''
        return tuple(self.attrs['voxel_origin'])

    @property
    def bounding_box(self):
        """
        Image bounding box in voxel units.

        Returns the image bounding box, defined by the minimum and maximum voxel
        coordinates in each spatial direction (x, y, z). The bounding box is
        calculated based on the voxel origins and lengths.

        Returns
        -------
        tuple
            A tuple containing two tuples: ((xmin, ymin, zmin), (xmax, ymax, zmax)),
            namely the minimum and maximum coordinates in the x, y, and z directions.
        """
        return    ((self.ox, self.oy, self.oz),
                   (self.ox + (self.nx-1)*self.hx,
                    self.oy + (self.ny-1)*self.hy,
                    self.oz + (self.nz-1)*self.hz))

    def chunk_slice_indices(self, chunk_id):
        """
        Calculate the slice indices for a given Zarr chunk.

        This method computes the first and last+1 indices for a specific Zarr chunk
        within the array, based on the block's ID. Useful for working with chunked
        data in parallel processing.

        Parameters
        ----------
        chunk_id : int
            The ID of the block for which to calculate the slice indices.

        Returns
        -------
        tuple
            A tuple containing six integers: (box, bex, boy, bey, boz, bez).
            These represent the start and end indices for the block in the
            x, y, and z directions, respectively.
        """
        if chunk_id >= self.nchunks:
            raise ValueError(f'chunk_id={chunk_id}: array has only {self.nchunks} chunks.')
        Nblocks = self.cdata_shape
        bk = chunk_id // (Nblocks[0]*Nblocks[1])
        bj = (chunk_id - bk*Nblocks[0]*Nblocks[1]) // Nblocks[0]
        bi = chunk_id - bk*Nblocks[0]*Nblocks[1] - bj*Nblocks[0]

        box = bi*self.chunks[0]
        bex = min(box+self.chunks[0], self.shape[0])
        boy = bj*self.chunks[1]
        bey = min(boy+self.chunks[1], self.shape[1])
        boz = bk*self.chunks[2]
        bez = min(boz+self.chunks[2], self.shape[2])

        return box, bex, boy, bey, boz, bez

    def get_voxel_coordinates(self, i, j, k):
        """
        Get the spatial coordinates of the voxel at a given position.

        This method calculates the spatial coordinates of the voxel located at
        the specified (i, j, k) position, taking into account
        the voxel origin and voxel length.

        Parameters
        ----------
        i : int
            The voxel index in the x-direction, 0 <= i < nx.
        j : int
            The voxel index in the y-direction, 0 <= j < ny.
        k : int
            The voxel index in the z-direction, 0 <= k < nz.

        Returns
        -------
        tuple
            A tuple containing the spatial coordinates (x, y, z) of the specified voxel.
        """
        return (self.ox+i*self.hx, self.oy+j*self.hy, self.oz+k*self.hz)

    def get_closest_voxel_index(self, x, y, z, allow_outside=False):
        """
        Get the voxel index closest to a given spatial position.

        This method calculates the (i, j, k) indices of the image voxel closest
        to the specified spatial coordinates (x, y, z).

        Parameters
        ----------
        x : float
            The spatial coordinate in the x-direction.
        y : float
            The spatial coordinate in the y-direction.
        z : float
            The spatial coordinate in the z-direction.
        allow_outside : Boolean
            If False (default) return None in case the spatial coordinates point
            to a region outside the image bounding box.

        Returns
        -------
        tuple or None
            A tuple containing the voxel indices (i, j, k) corresponding to the
            specified spatial coordinates if the point (x, y, z) is inside the
            image, or None otherwise.
        """
        pos = (int(round((x-self.ox)/self.hx)),
               int(round((y-self.oy)/self.hy)),
               int(round((z-self.oz)/self.hz)))
        if 0<=pos[0]<self.nx and 0<=pos[1]<=self.ny and 0<=pos[2]<=self.nz:
            return pos
        elif allow_outside:
            return pos
        return None


    def check_mask_and_segmentation(self, *, mask=None, segmentation=None):
        """
        Validate mask and segmentation compatibility.

        This method checks the validity of the provided mask, segmentation, and
        phases to ensure they are compatible with the voxel image.

        Parameters
        ----------
        mask : VoxelImage, optional
            A mask voxel image. If provided, must be a voxel image of boolean
            dtype with the same shape, voxel length, voxel origin, and voxel
            unit.
        segmentation : VoxelImage, optional
            A segmentation voxel image. If provided, must be a voxel image of
            boolean or unsigned integer dtype with the same shape, voxel length,
            voxel origin, and voxel unit.

        Raises
        ------
        ValueError
            If any of the provided parameters are invalid or incompatible with
            the current voxel image.
        """
        return
        if mask is not None:
            _assert.rockverse_instance(mask, 'mask', ('VoxelImage',))
            _assert.dtype('mask', mask, 'boolean', 'b')
            _assert.same_shape('Mask image', (mask, self))
            _assert.same_voxel_length('Mask image', (mask, self))
            _assert.same_voxel_origin('Mask image', (mask, self))
            _assert.same_voxel_unit('Mask image', (mask, self))
        if segmentation is not None:
            _assert.rockverse_instance(segmentation, 'segmentation', ('VoxelImage',))
            _assert.dtype('Segmentation', segmentation, 'boolean or integer', 'bui')
            _assert.same_shape('Segmentation image', (segmentation, self))
            _assert.same_voxel_length('Segmentation image', (segmentation, self))
            _assert.same_voxel_origin('Segmentation image', (segmentation, self))
            _assert.same_voxel_unit('Segmentation image', (segmentation, self))

    def math(self, a, op, *, mask=None, segmentation=None, phases=(),
             region=None, use_gpu=True):
        '''
        :bdg-info:`Parallel`
        :bdg-info:`CPU`
        :bdg-info:`GPU`

        Parallel computing element-wise math operations.

        This method applies math operations for each voxel in the image.
        Optional parameters allow for selective setting based on mask,
        segmentation, and phases.

        Parameters
        ----------
        a : scalar
            The value to be used in the operation.

            .. note::
                There is no internal check for the correct data type of ``a``.
                Ensure that the ``a`` provided is compatible with the image
                data type.

        op : str
            The operation to be performed. Let :math:`v` represent each voxel
            in the image. Valid values for ``op`` are:

            .. list-table::
                :header-rows: 1

                * -  ``op``
                  - Operation
                * - 'set'
                  - :math:`v = a`
                * - 'add'
                  - :math:`v = v + a`
                * - 'subtract'
                  - :math:`v = v - a`
                * - 'multiply'
                  - :math:`v = v \\times a`
                * - 'divide'
                  - :math:`v = v / a`
                * - 'logical and'
                  - :math:`v = v \\land a`
                * - 'logical or'
                  - :math:`v = v \\lor a`
                * - 'logical xor'
                  - :math:`v = v \\oplus a` (exclusive OR)
                * - 'min'
                  - :math:`v = \\min(v, a)`
                * - 'max'
                  - :math:`v = \\max(v, a)`

        mask : voxel image, optional
            Boolean voxel image. If provided, the operation will ignore masked
            voxels (i.e., where mask is True).
        segmentation : voxel image, optional
            Unsigned integer voxel image. If provided, only voxels where the
            segmentation phase is in ``phases`` will be set to the specified
            value.
        phases : iterable of int, optional
            Any iterable with non negative integers representing the
            segmentation phases. Used together with ``segmentation``.
            Segmentation phases not in ``phases`` will be ignored by the
            operation.
        region : Region, optional
            A region specification. If provided, only voxels within the
            specified region will be set to the specified value.
        use_gpu : bool, optional
            Whether to use available GPUs for computation. Default is True.

        '''
        self.check_mask_and_segmentation(mask=mask, segmentation=segmentation)
        _assert.iterable.any_iterable_non_negative_integers('phases', phases)
        _array_math(array1=self,
                    array2=None,
                    value=a,
                    op=op,
                    mask=mask,
                    segmentation=segmentation,
                    phases=phases,
                    region=region,
                    use_gpu=use_gpu)


    def combine(self, a, op, *, mask=None, segmentation=None, phases=(),
             region=None, use_gpu=True):
        '''
        :bdg-info:`Parallel`
        :bdg-info:`CPU`
        :bdg-info:`GPU`

        Combine another voxel image by element-wise math operations.

        This method applies math operations for each voxel in the images.
        Optional parameters allow for selective setting based on mask,
        segmentation, and phases.

        Parameters
        ----------
        a : VoxelImage
            The voxel image to be used in the operation. Must have shape,
            voxel origin, voxel length and voxel unit equal to the ones in the
            original array.

            .. note::
                There is no internal check for the correct data type of ``a``.
                Ensure that the ``a`` provided is compatible with the image
                data type.

        op : str
            The operation to be performed. Let :math:`v` represent each voxel
            in the original image and :math:`a` represent each voxel in ``a``.
            Valid values for ``op`` are:

            .. list-table::
                :header-rows: 1

                * -``op``
                  - Operation
                * - 'copy'
                  - :math:`v = a`
                * - 'add'
                  - :math:`v = v + a`
                * - 'subtract'
                  - :math:`v = v - a`
                * - 'multiply'
                  - :math:`v = v \\times a`
                * - 'divide'
                  - :math:`v = v / a`
                * - 'logical and'
                  - :math:`v = v \\land a`
                * - 'logical or'
                  - :math:`v = v \\lor a`
                * - 'logical xor'
                  - :math:`v = v \\oplus a` (exclusive OR)
                * - 'min'
                  - :math:`v = \\min(v, a)`
                * - 'max'
                  - :math:`v = \\max(v, a)`
                * - 'average'
                  - :math:`v = (v + a)/2`
                * - 'absolute difference'
                  - :math:`v = |v - a|`

        mask : VoxelImage, optional
            Boolean voxel image. If provided, the operation will ignore masked
            voxels (i.e., where mask is True).
        segmentation : VoxelImage, optional
            Unsigned integer voxel image. If provided, only voxels where the
            segmentation phase is in ``phases`` will be set to the specified
            value.
        phases : iterable of int, optional
            Any iterable with non negative integers representing the
            segmentation phases. Used together with ``segmentation``.
            Segmentation phases not in ``phases`` will be ignored by the
            operation.
        phases : iterable of int, optional
            Any iterable of non negative integers representing the phases. Used
            together with ``segmentation`` to specify which segmentation phases
            should be set to the specified value.
        region : Region, optional
            A region specification. If provided, only voxels within the
            specified region will be set to the specified value.
        use_gpu : bool, optional
            Whether to use available GPUs for computation. Default is True.

        '''
        self.check_mask_and_segmentation(mask=mask, segmentation=segmentation)
        _assert.iterable.any_iterable_non_negative_integers('phases', phases)
        _array_math(array1=self,
                    array2=a,
                    value=None,
                    op=op,
                    mask=mask,
                    segmentation=segmentation,
                    phases=phases,
                    region=region,
                    use_gpu=use_gpu)



    def export_raw(self, filename, dtype=None, order='F', byteorder='=',
                   write_fiji_macro=False):
        '''
        :bdg-info:`Parallel`
        :bdg-info:`CPU`

        Export a voxel image to a raw file and corresponding metadata to a JSON
        file.

        This method exports the voxel image data into a raw binary file
        specified by `filename`. Corresponding metadata is saved in a JSON file
        with the same name as the raw file but with a `.json` extension.

        Parameters
        ----------
        filename : str
            Path to the exported raw file in the file system.
        dtype : {None, Numpy dtype}, optional
            Data type to be used in the exported raw file. If None, the original
            image data type will be used. Boolean types will be cast to unsigned
            8-bit integers in the exported file.
            .. warning::
                Pay attention if setting `dtype`, as there is no internal check
                to ensure that the exported data will be correctly cast from the
                original array dtype.
        order : {'C', 'F'}, optional
            The order of the exported raw array layout: 'C' for C-style (most
            rapidly changing index last), 'F' for Fortran-style (most rapidly
            changing index first). Use 'F' when exporting to Fiji/ImageJ.
            Default is 'F'.
        byteorder : {'<', '>', '='}, optional
            Byte order for the exported raw data. This parameter has precedence
            over the ``dtype`` parameter. Use ``'<'`` to force little-endian,
            ``'>'`` to force big-endian, or ``'='`` to keep the original byte
            order. Default is ``'='``.
        write_fiji_macro : bool, optional
            If True, writes a convenience macro file (<filename>.ijm) for
            opening the exported raw file in ImageJ/Fiji. In this case, byte
            size in ``dtype`` must be up to 64 bits, and array order must be
            in Fortran-style. Default is False.

        Raises
        ------
        ValueError
            If the array order is not 'F' when `write_fiji_macro` is True.
            If the byte size in `dtype` exceeds 64 bits when `write_fiji_macro`
            is True.
        '''
        _assert.instance('filename', filename, 'string', (str,))
        if dtype is not None:
            try:
                dtypeout = np.dtype(dtype)
            except Exception as e:
                _assert.collective_raise(e)
        else:
            dtypeout = self.dtype
        dtypeout = dtypeout.str
        _assert.in_group('order', order, ('C', 'F'))
        _assert.in_group('byteorder', byteorder, ('<', '>', '='))
        if byteorder in '<>':
            dtypeout = byteorder + dtypeout[1:]
        _assert.instance('write_fiji_macro', write_fiji_macro, 'boolean', (bool,))
        if write_fiji_macro:
            if order != 'F':
                _assert.collective_raise(ValueError(
                    "Exported raw data must be in Fortran order when requesting ImageJ/Fiji macro."))
            if int(dtypeout[2:]) > 8:
                _assert.collective_raise(ValueError(
                    "ImageJ/Fiji macro can't be written for data types larger than 64 bits."))

        attrs = {'description': self.attrs['description'],
                 'shape': tuple(self.shape),
                 'voxel length': tuple(self.attrs['voxel_length']),
                 'voxel_unit': self.attrs['voxel_unit'],
                 'voxel_origin': tuple(self.attrs['voxel_origin']),
                 'array_order': order,
                 'Numpy dtype': dtypeout,
                 }

        if dtypeout[1] == 'b':
            attrdtype = 'boolean'
            dtypeout = '|b1'
        else:
            attrdtype = f'{int(dtypeout[2:])*8}-bit'
            if dtypeout[1] == 'u':
                attrdtype = f'{attrdtype} unsigned integer'
            elif dtypeout[1] == 'i':
                attrdtype = f'{attrdtype} signed integer'
            elif dtypeout[1] == 'f':
                attrdtype = f'{attrdtype} float'
            if int(dtypeout[2:]) > 1:
                if dtypeout[0] == '<':
                    attrdtype = f'{attrdtype}, little endian'
                elif dtypeout[0] == '>':
                    attrdtype = f'{attrdtype}, big endian'
        attrs['data type'] = attrdtype

        if mpi_rank == 0:
            file = np.memmap(filename, dtype=dtypeout, mode='w+',
                             shape=self.shape, order=order)
            file.flush()
        comm.barrier()
        file = np.memmap(filename, dtype=dtypeout, mode='r+',
                         shape=self.shape, order=order)
        for chunk_id in rvtqdm(range(self.nchunks), desc='Exporting raw file', unit='chunk'):
            if chunk_id % mpi_nprocs == mpi_rank:
                box, bex, boy, bey, boz, bez = self.chunk_slice_indices(chunk_id)
                file[box:bex, boy:bey, boz:bez] = self[box:bex, boy:bey, boz:bez]
        comm.barrier()
        file.flush()
        comm.barrier()
        if mpi_rank == 0:
            with open(filename+'.json', 'w') as fp:
                json.dump(attrs, fp, indent=4)
        comm.barrier()
        if write_fiji_macro:
            cmd = f"//{attrs['description']}\n" if attrs['description'] else ''
            cmd = f'{cmd}run("Raw...", "open={filename} image='
            if int(dtypeout[2:]) == 1:
                cmd = f'{cmd}8-bit '
            else:
                cmd = f'{cmd}[{int(dtypeout[2:])*8}-bit '
                if dtypeout[1] == 'u':
                    cmd = f'{cmd}Unsigned] '
                elif dtypeout[1] == 'i':
                    cmd = f'{cmd}Signed] '
                elif dtypeout[1] == 'f':
                    cmd = f'{cmd}Real] '
            cmd = f'{cmd}width={self.shape[0]} height={self.shape[1]} number={self.shape[2]}'
            if int(dtypeout[2:]) >1 and dtypeout[0] == '<':
                cmd = f'{cmd} little-endian'
            cmd = f'{cmd}");\n'
            if mpi_rank == 0:
                with open(filename+'.ijm', 'w') as fp:
                    fp.write(cmd);
        comm.barrier()


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
        The created voxel image with the specified shape and metadata.
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
        The created empty voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
        The created voxel image.
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
                GPU=True,
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
        The created voxel image representing the sphere pack.
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
                      fill_value=fill_value, GPU=GPU)
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
               store=None,
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
    store : str or Zarr Store, optional
        Zarr store or path to a directory in the file system, or name of a zip
        file to be passed to the Zarr array creation function. Default is None.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        :ref:`creation function <rockverse_digitalrock_create_function>`.

    Returns
    -------
    VoxelImage
        The created voxel image with the imported raw data.
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
    _assert.dtype('dtype', dtype)
    _assert.instance('name', field_name, 'string', (str,))
    _assert.instance('unit', field_unit, 'string', (str,))
    _assert.instance('offset', offset, 'integer', (int,))
    _assert.in_group('raw_file_order', raw_file_order, ('C', 'F'))
    _assert.instance('voxel_unit', voxel_unit, 'string', (str,))
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))
    _assert.zarr_store('store', store)

    kwargs['dtype'] = np.dtype(dtype).kind + str(np.dtype(dtype).itemsize)
    kwargs['description'] = description
    kwargs['field_name'] = field_name
    kwargs['field_unit'] = field_unit
    kwargs['shape'] = shape
    kwargs['store'] = store
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