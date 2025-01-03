"""
The histogram module provides the Histogram class for computing and analyzing
histograms, probability density functions (PDFs), cumulative distribution
functions (CDFs), and percentiles from VoxelImage objects.

This module supports advanced features such as masked regions, regions of
interest (ROI), and segmentation-based phase analysis, enabling efficient voxel
data exploration.

.. todo::
    * Improve handling of step changes in quantile functions when calculating percentiles.
    * Put countter simply as int
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
import warnings
from numba import njit
from rockverse import _assert
from rockverse._utils import rvtqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

@njit()
def _apply_mask_cpu(skip, mask):
    nx, ny, nz = skip.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    skip[i, j, k] = True

@njit()
def _block_count_phases(block_segm, count):
    #Detect if nth phase is present in segmentation
    nx, ny, nz = block_segm.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if count[block_segm[i, j, k]] == 0:
                    count[block_segm[i, j, k]] = 1

@njit()
def _block_update_histogram(hist, block_data, block_segm, skip, bins, phases):
    nx, ny, nz = block_data.shape
    hist_shape = hist.shape
    num_phases = hist_shape[1] - 1
    N = len(bins)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k] or block_data[i, j, k] < bins[0] or block_data[i, j, k] > bins[-1]:
                    continue
                value = block_data[i, j, k]
                for ind in range(1, N):
                    if value < bins[ind]:
                        hist[ind-1, 0] += 1
                        if num_phases > 0:
                            phase = block_segm[i, j, k]
                            for p, ph in enumerate(phases):
                                if phase == ph:
                                    hist[ind-1, p + 1] += 1
                                    break
                        break

class Histogram():
    '''
    :bdg-info:`Parallel`
    :bdg-info:`CPU`

    Compute and manage histograms, probability density functions (PDFs),
    cumulative distribution functions (CDFs), and percentiles for VoxelImage data.

    This class provides methods to calculate histograms, PDFs, CDFs, and
    percentiles, with support for regions of interest, masks, and segmentation.

    Parameters
    ----------
    image : VoxelImage
        Input image.

    bins : int or sequence of scalars, optional
        If int, number of equal-width bins in the range of image min and max.
        If sequence, defines bin edges (including rightmost edge), and ignores
        values outside bins when calculating the histogram.
        Default value is 256.

    region : Region (optional)
        The region of interest in the image. If specified, only voxels within
        the region will be considered when computing the histogram.

    mask : Boolean VoxelImage (optional)
        The mask to apply on the image data. If specified, only unmasked
        voxels will be considered when computing the histogram.

    segmentation : Unsigned integer VoxelImage (optional)
        The segmentation data for image regions. If specified, histograms will
        also have individual counts for each segmentation phase.
    '''

    def __init__(self, image, *, bins=None, region=None, mask=None, segmentation=None):

        _assert.rockverse_instance(image, 'image', ('VoxelImage',))
        if region is not None:
            _assert.rockverse_instance(region, 'region', ('Region',))

        self._image = image
        self._image.check_mask_and_segmentation(mask=mask, segmentation=segmentation)
        self._mask = mask
        self._segmentation = segmentation
        self._region = region
        self._input_bins = bins
        self._min = None
        self._max = None
        self._bins = None
        self._phases = None
        self._hist = None

        self._get_min_max()
        self._update_bins()
        self._count_phases()
        self._update_histogram()


    def _get_min_max(self):
        if self._image.dtype.kind == 'b': #Boolean image? Treat as 0s and 1s
            self._min = 0
            self._max = 1
            return
        local_min = None
        local_max = None
        if mpi_rank == 0:
            local_min = self._image[0, 0, 0]
            local_max = self._image[0, 0, 0]
        local_min = comm.bcast(local_min, root=0)
        local_max = comm.bcast(local_max, root=0)
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        for block_id in rvtqdm(range(self._image.nchunks), desc=f'Histogram {self._image.field_name} (min/max)', unit='chunk'):
            if block_id % mpi_nprocs != mpi_rank:
                continue
            box, bex, boy, bey, boz, bez = self._image.chunk_slice_indices(block_id)
            block_data = self._image[box:bex, boy:bey, boz:bez]
            skip = np.zeros((bex-box, bey-boy, bez-boz), dtype='bool')
            if self._mask is not None:
                _apply_mask_cpu(skip, self._mask[box:bex, boy:bey, boz:bez])
            if self._region is not None:
                self._region.mask_chunk_cpu(skip, ox, oy, oz, hx, hy, hz, box, boy, boz)
            block_data = ma.masked_array(block_data, mask=skip)
            minblock = block_data.min()
            if minblock < local_min:
                local_min = minblock
            maxblock = block_data.max()
            if maxblock > local_max:
                local_max =  maxblock
        comm.barrier()
        self._min = comm.allreduce(local_min, op=MPI.MIN)
        self._max = comm.allreduce(local_max, op=MPI.MAX)


    def _update_bins(self):
        v = self._input_bins
        if self._image.dtype.kind == 'b': #Boolean image? Treat as 0's and 1's
            self._bins = np.array([0, 1])
            return
        if v is None:
            self._bins = np.linspace(start=self._min,
                                     stop=self._max,
                                     num=256,
                                     dtype=self._image.dtype)
        #Positive integer
        elif np.dtype(type(v)).kind in 'ui' and v>0:
            self._bins = np.linspace(start=self._min,
                                     stop=self._max,
                                     num=v,
                                     dtype=self._image.dtype)
        #Ordered iterable of numbers
        elif (hasattr(v, '__iter__') and hasattr(v, '__getitem__')
              and all(np.dtype(type(k)).kind in 'uif' for k in v)):
            self._bins = np.sort(v)
        else:
            if mpi_rank == 0:
                warnings.warn('Invalid value for bins. Falling back to default.')
            self._input_bins = None
            self._update_bins()


    def _count_phases(self):
        if self._segmentation is None:
            self._phases = ()
            return
        count = np.zeros(2**(8*self._segmentation.dtype.itemsize), dtype=int)
        for block_id in rvtqdm(range(self._image.nchunks), desc=f'Histogram {self._image.field_name} (reading segmentation)', unit='chunk'):
            if block_id % mpi_nprocs != mpi_rank:
                continue
            box, bex, boy, bey, boz, bez = self._image.chunk_slice_indices(block_id)
            block_segm = self._segmentation[box:bex, boy:bey, boz:bez]
            _block_count_phases(block_segm, count)
        comm.barrier()
        #All-reduce phases
        gphases = set()
        for rank in range(mpi_nprocs):
            lph = comm.bcast(np.argwhere(count>0).flatten(), root=rank)
            gphases = gphases.union(set(lph))
        self._phases = np.sort(list(gphases)).astype(self._segmentation.dtype)


    def _update_histogram(self):
        hist = np.zeros((len(self._bins)-1, len(self._phases)+1), dtype=int)
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        for block_id in rvtqdm(range(self._image.nchunks), desc=f'Histogram {self._image.field_name} (counting voxels)', unit='chunk'):
            if block_id % mpi_nprocs != mpi_rank:
                continue
            box, bex, boy, bey, boz, bez = self._image.chunk_slice_indices(block_id)
            block_data = self._image[box:bex, boy:bey, boz:bez]
            if block_data.dtype.kind == 'b': #Boolean?
                block_data = block_data.astype('u1')
            skip = np.zeros((bex-box, bey-boy, bez-boz), dtype='bool')
            if self._mask is not None:
                _apply_mask_cpu(skip, self._mask[box:bex, boy:bey, boz:bez])
            if self._region is not None:
                self._region.mask_chunk_cpu(skip, ox, oy, oz, hx, hy, hz, box, boy, boz)
            if len(self._phases)>0:
                block_segm = self._segmentation[box:bex, boy:bey, boz:bez]
                phases = self._phases
            else:
                block_segm = skip
                phases = np.array([0])
            _block_update_histogram(hist, block_data, block_segm, skip, self._bins, phases)
        comm.barrier()
        #All-reduce counts
        for c in range(hist.shape[1]):
            hist[:, c] = comm.allreduce(hist[:, c], op=MPI.SUM)
        self._hist = pd.DataFrame(hist, columns=['full',]+list(self._phases))


    @property
    def image(self):
        '''
        The input voxel image.
        '''
        return self._image

    @property
    def region(self):
        """
        The region associated to the histogram.
        """
        return self._region

    @property
    def mask(self):
        """
        The mask associated to the histogram.
        """
        return self._mask

    @property
    def segmentation(self):
        """
        The segmentation associated to the histogram.
        """
        return self._segmentation

    @property
    def phases(self):
        '''
        Tuple with segmentation phases.
        '''
        return tuple(self._phases)

    @property
    def bins(self):
        """
        The histogram bins.
        """
        return self._bins

    @property
    def bin_centers(self):
        """
        The centers of the histogram bins.
        """
        return (self._bins[1:].astype(float)+self._bins[:-1].astype(float))/2

    @property
    def min(self):
        """
        The image minimum value.
        """
        return self._min

    @property
    def max(self):
        """
        The image maximum value.
        """
        return self._max

    @property
    def count(self):
        """
        Get the histogram count as a Pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame`
            A data frame containing the histogram values for the full image and each segmentation phase.
        """
        return self._hist

    @property
    def pdf(self):
        """
        Compute the Probability Density Function (PDF) from the calculated
        histogram. PDF values are normalized such that the total area for the
        bins (bin width times histogram height) equals 1.

        Returns
        -------
        pdf : pandas.DataFrame
            DataFrame containing the PDF values for the full image and for each
            segmentation phase.
        """
        h = self._hist.copy(deep=True)
        x = self._bins.astype(float)
        y = self._hist['full'].values.astype(float)
        area = 0.
        for k, yi in enumerate(y):
            area += (x[k+1]-x[k])*yi
        h /= area
        return h

    @property
    def cdf(self):
        """
        Compute the Cumulative Distribution Function (CDF) for the full image and
        phase by phase as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the CDF values for the full image and each
            segmentation phase.

        Examples
        --------
            >>> # Get the CDF DataFrame
            >>> cdf = histogram.cdf
            >>> # Access the CDF values for a specific segmentation phase
            >>> phase_cdf = cdf[phase_id]
        """

        x = self.bins.astype(float)
        pdf = self.pdf
        h = pdf*0.
        h.loc[len(h)] = 0.
        for k in range(1, len(h)):
            h.loc[k] = h.loc[k-1] + pdf.loc[k-1]*(x[k]-x[k-1])
        return h


    def percentile(self, q):
        """
        Estimate the q-th percentiles by linear interpolation on
        histogram CDF and bins.

        Parameters
        ----------
        q : float or array-like of floats
            Percentage or sequence of percentages. Must obey 0<=q<=100.

        Returns
        -------
        float or array-like of floats
            Percentile values.

        Examples
        --------
            >>> # Compute the 10th percentile
            >>> p = histogram.percentile(10)
            >>> # Compute quartiles (25th, 50th, and 75th percentiles)
            >>> Q1, Q2, Q3 = histogram.percentile([25, 50, 75])
        """
        x = self.cdf['full'].values.astype(float)
        y = self.bins
        if not (
            (np.dtype(type(q)).kind in 'uif' and 0<=q<=100)
            or
            (hasattr(q, '__iter__') and hasattr(q, '__getitem__')
             and all(np.dtype(type(k)).kind in 'uif' for k in q)
             and all(0<=k<=100 for k in q)
             )
             ):
            _assert.collective_raise(ValueError(
                'q must be a number of a iterable of numbers between 0 and 100.'))
        return np.interp(np.array(q)/100, x, y)
