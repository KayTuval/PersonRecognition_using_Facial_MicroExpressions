#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import numpy
import math

from bob.bio.base.extractor import Extractor


class LGBPHS(Extractor):
    """Extracts *Local Gabor Binary Pattern Histogram Sequences* (LGBPHS) [ZSG+05]_ from the images, using functionality from :ref:`bob.ip.base <bob.ip.base>` and :ref:`bob.ip.gabor <bob.ip.gabor>`.

    The block size and the overlap of the blocks can be varied, as well as the parameters of the Gabor wavelet (:py:class:`bob.ip.gabor.Transform`) and the LBP extractor (:py:class:`bob.ip.base.LBP`).

    **Parameters:**

    block_size : int or (int, int)
      The size of the blocks that will be extracted.
      This parameter might be either a single integral value, or a pair ``(block_height, block_width)`` of integral values.

    block_overlap : int or (int, int)
      The overlap of the blocks in vertical and horizontal direction.
      This parameter might be either a single integral value, or a pair ``(block_overlap_y, block_overlap_x)`` of integral values.
      It needs to be smaller than the ``block_size``.

    gabor_directions, gabor_scales, gabor_sigma, gabor_maximum_frequency, gabor_frequency_step, gabor_power_of_k, gabor_dc_free
      The parameters of the Gabor wavelet family, with its default values set as given in [WFK97]_.
      Please refer to :py:class:`bob.ip.gabor.Transform` for the documentation of these values.

    use_gabor_phases : bool
      Extract also the Gabor phases (inline) and not only the absolute values.
      In this case, Extended LGBPHS features [ZSQ+09]_ will be extracted.

    lbp_radius, lbp_neighbor_count, lbp_uniform, lbp_circular, lbp_rotation_invariant, lbp_compare_to_average, lbp_add_average
      The parameters of the LBP.
      Please see :py:class:`bob.ip.base.LBP` for the documentation of these values.

      .. note::
         The default values are as given in [ZSG+05]_ (the values of [ZSQ+09]_ might differ).

    sparse_histogram : bool
      If specified, the histograms will be handled in a sparse way.
      This reduces the size of the extracted features, but the computation will take longer.

      .. note::
         Sparse histograms are only supported, when ``split_histogram = None``.

    split_histogram : one of ``('blocks', 'wavelets', 'both')`` or ``None``
      Defines, how the histogram sequence is split.
      This could be interesting, if the histograms should be used in another way as simply concatenating them into a single histogram sequence (the default).
    """

    def __init__(
            self,
            # Block setup
            block_size,  # one or two parameters for block size
            block_overlap=0,  # one or two parameters for block overlap
            # Gabor parameters
            gabor_directions=8,
            gabor_scales=5,
            gabor_sigma=2. * math.pi,
            gabor_maximum_frequency=math.pi / 2.,
            gabor_frequency_step=math.sqrt(.5),
            gabor_power_of_k=0,
            gabor_dc_free=True,
            use_gabor_phases=False,
            # LBP parameters
            lbp_radius=2,
            lbp_neighbor_count=8,
            lbp_uniform=True,
            lbp_circular=True,
            lbp_rotation_invariant=False,
            lbp_compare_to_average=False,
            lbp_add_average=False,
            # histogram options
            sparse_histogram=False,
            split_histogram=None
    ):
        # call base class constructor
        Extractor.__init__(
            self,

            block_size=block_size,
            block_overlap=block_overlap,
            gabor_directions=gabor_directions,
            gabor_scales=gabor_scales,
            gabor_sigma=gabor_sigma,
            gabor_maximum_frequency=gabor_maximum_frequency,
            gabor_frequency_step=gabor_frequency_step,
            gabor_power_of_k=gabor_power_of_k,
            gabor_dc_free=gabor_dc_free,
            use_gabor_phases=use_gabor_phases,
            lbp_radius=lbp_radius,
            lbp_neighbor_count=lbp_neighbor_count,
            lbp_uniform=lbp_uniform,
            lbp_circular=lbp_circular,
            lbp_rotation_invariant=lbp_rotation_invariant,
            lbp_compare_to_average=lbp_compare_to_average,
            lbp_add_average=lbp_add_average,
            sparse_histogram=sparse_histogram,
            split_histogram=split_histogram
        )

        # block parameters
        self.block_size = block_size if isinstance(block_size, (tuple, list)) else (block_size, block_size)
        self.block_overlap = block_overlap if isinstance(block_overlap, (tuple, list)) else (
        block_overlap, block_overlap)
        if self.block_size[0] < self.block_overlap[0] or self.block_size[1] < self.block_overlap[1]:
            raise ValueError("The overlap is bigger than the block size. This won't work. Please check your setup!")

        # Gabor wavelet transform class
        self.gwt = bob.ip.gabor.Transform(
            number_of_scales=gabor_scales,
            number_of_directions=gabor_directions,
            sigma=gabor_sigma,
            k_max=gabor_maximum_frequency,
            k_fac=gabor_frequency_step,
            power_of_k=gabor_power_of_k,
            dc_free=gabor_dc_free
        )
        self.trafo_image = None
        self.use_phases = use_gabor_phases

        self.lbp = bob.ip.base.LBP(
            neighbors=lbp_neighbor_count,
            radius=float(lbp_radius),
            circular=lbp_circular,
            to_average=lbp_compare_to_average,
            add_average_bit=lbp_add_average,
            uniform=lbp_uniform,
            rotation_invariant=lbp_rotation_invariant,
            border_handling='wrap'
        )

        self.split = split_histogram
        self.sparse = sparse_histogram
        if self.sparse and self.split:
            raise ValueError("Sparse histograms cannot be split! Check your setup!")

    def _fill(self, lgbphs_array, lgbphs_blocks, j):
        """Copies the given array into the given blocks"""
        # fill array in the desired shape
        if self.split is None:
            start = j * self.n_bins * self.n_blocks
            for b in range(self.n_blocks):
                lgbphs_array[start + b * self.n_bins: start + (b + 1) * self.n_bins] = lgbphs_blocks[b][:]
        elif self.split == 'blocks':
            for b in range(self.n_blocks):
                lgbphs_array[b, j * self.n_bins: (j + 1) * self.n_bins] = lgbphs_blocks[b][:]
        elif self.split == 'wavelets':
            for b in range(self.n_blocks):
                lgbphs_array[j, b * self.n_bins: (b + 1) * self.n_bins] = lgbphs_blocks[b][:]
        elif self.split == 'both':
            for b in range(self.n_blocks):
                lgbphs_array[j * self.n_blocks + b, 0: self.n_bins] = lgbphs_blocks[b][:]

    def _sparsify(self, array):
        """This function generates a sparse histogram from a non-sparse one."""
        if not self.sparse:
            return array
        if len(array.shape) == 2 and array.shape[0] == 2:
            # already sparse
            return array
        assert len(array.shape) == 1
        indices = []
        values = []
        for i in range(array.shape[0]):
            if array[i] != 0.:
                indices.append(i)
                values.append(array[i])
        return numpy.array([indices, values], dtype=numpy.float64)

    def __call__(self, image):
        """__call__(image) -> feature

        Extracts the local Gabor binary pattern histogram sequence from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
          The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
          The list of Gabor jets extracted from the image.
          The 2D location of the jet's nodes is not returned.
        """ 
        """"""
        assert image.ndim == 2
        assert isinstance(image, numpy.ndarray)
        assert image.dtype == numpy.float64

        # perform GWT on image
        if self.trafo_image is None or self.trafo_image.shape[1:3] != image.shape:
            # create trafo image
            self.trafo_image = numpy.ndarray((self.gwt.number_of_wavelets, image.shape[0], image.shape[1]),
                                             numpy.complex128)

        # perform Gabor wavelet transform
        self.gwt.transform(image, self.trafo_image)

        jet_length = self.gwt.number_of_wavelets * (2 if self.use_phases else 1)

        lgbphs_array = None 
        # iterate through the layers of the trafo image
        for j in range(self.gwt.number_of_wavelets):
            # compute absolute part of complex response
            abs_image = numpy.abs(self.trafo_image[j])
            # Computes LBP histograms
            abs_blocks = bob.ip.base.lbphs(abs_image, self.lbp, self.block_size, self.block_overlap)

            # Converts to Blitz array (of different dimensionalities)
            self.n_bins = abs_blocks.shape[1]
            self.n_blocks = abs_blocks.shape[0]

            if self.split is None:
                shape = (self.n_blocks * self.n_bins * jet_length,)
            elif self.split == 'blocks':
                shape = (self.n_blocks, self.n_bins * jet_length)
            elif self.split == 'wavelets':
                shape = (jet_length, self.n_bins * self.n_blocks)
            elif self.split == 'both':
                shape = (jet_length * self.n_blocks, self.n_bins)
            else:
                raise ValueError("The split parameter must be one of ['blocks', 'wavelets', 'both'] or None")

            # create new array if not done yet
            if lgbphs_array is None:
                lgbphs_array = numpy.ndarray(shape, 'float64')

            # fill the array with the absolute values of the Gabor wavelet transform
            self._fill(lgbphs_array, abs_blocks, j)

            if self.use_phases:
                # compute phase part of complex response
                phase_image = numpy.angle(self.trafo_image[j])
                # Computes LBP histograms
                phase_blocks = bob.ip.base.lbphs(phase_image, self.lbp, self.block_size, self.block_overlap)
                # fill the array with the phases at the end of the blocks
                self._fill(lgbphs_array, phase_blocks, j + self.gwt.number_of_wavelets)

        # return the concatenated list of all histograms
        return self._sparsify(lgbphs_array)

    # re-define the train function to get it non-documented


def train(*args, **kwargs): raise NotImplementedError("This function is not implemented and should not be called.")


def load(*args, **kwargs): pass
