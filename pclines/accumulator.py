"""
The module provides the class PCLines with "low level" functions for
* accumulation of observations to PCLines space,
* point mapping from PCLines space to homogeneous lines
that can be used to construct a custom PCLines transform of user-defined
edge points (observations).


References
----------
[1] Dubska et al, PCLines - Line detection with parallel coordinates, CVPR 2011
"""


import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology.grey import erosion, dilation
from skimage.morphology import disk

from .rasterizer import polys


def _linear_transform(src, dst):
    """ Parameters of a linear transform from range specifications """
    (s0, s1), (d0,d1) = src, dst
    w = (d1 - d0) / (s1 - s0)
    b = d0 - w*s0
    return w, b


def _validate_points(x:np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Points must be numpy array")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Points must be 2D array with 2 columns")


def _validate_bbox(bbox):
    if not isinstance(bbox, (tuple, list, np.ndarray)):
        raise TypeError("bbox must be tuple, list or numpy array")
    if len(bbox) != 4:
        raise ValueError("bbox must contain 4 elements")
    x,y,w,h = np.array(bbox,"f")
    if w <= 0 or h <= 0:
        raise ValueError("bbox width and heigh must be positive")


class Normalizer:
    """
    Linear mapping of ranges

    The class maps values from source range (a,b) to a destination
    range (c,d) by scaling and offset.

    The `transform` then maps a to c and b to d. The `inverse`
    then does inverse transform.
    """
    def __init__(self, src_range, dst_range=(0,1)):
        """
        Inputs
        ------
        src_range, dst_range: tuple
            A tuple with two values [a, b] of source and destination ranges

        Example
        -------
        >> T = Normalizer((0,10), (1,-1))
        >> T.transform(10)
        -1
        >> T.transform(6)
        -0.2
        >> T.inverse(-1)
        10
        """
        # TODO: check ranges
        self.w, self.b = _linear_transform(src_range, dst_range)
        self.wi, self.bi = _linear_transform(dst_range, src_range)

    def transform(self, x):
        """ src -> dst mapping of x

        Transforms values of x from source range to destination range.
        
        Inputs
        ------
        x : ndarray
            values to transform

        Outputs
        -------
        y : ndarray
            Transformed x

        Example
        -------
        >> T = Normalizer((0, 100), (0, 1))
        >> T(50)
        0.5
        """
        return self.w * x + self.b

    def inverse(self, x):
        """ dst -> src mapping of x
        
        Transforms values of x from destination range to the source range.
        
        Inputs
        ------
        x : ndarray
            values to transform

        Outputs
        -------
        y : ndarray
            Transformed x

        Example
        -------
        >> T = Normalizer((0, 100), (0, 1))
        >> T(0.5)
        50
        """
        return self.wi * x + self.bi

    __call__ = transform


class PCLines:
    """
    Wrapper for PCLines accumulator of certain size
    """
    def __init__(self, bbox, d=256):
        _validate_bbox(bbox)
        self.bbox = bbox

        # Init accumulator
        shape = d, 2*d-1
        self.A = np.zeros(shape, "f")
        self.d = d

        ranges = d * np.array(self.input_shape)/self.scale
        ofs = (d-ranges) / 2
        (x0,y0),(x1,y1) = ofs, (ranges-1)+ofs

        x,y = self.origin
        w,h = self.input_shape
        self.norm_u = Normalizer((y,y+h+1), (y1, y0))
        self.norm_v = Normalizer((x,x+w+1), (x0, x1))
        self.norm_w = Normalizer((y,y+h+1), (y0, y1))

    @property
    def origin(self):
        """ The origin of the bounding box of the observations """
        return self.bbox[:2]

    @property
    def input_shape(self):
        """ Size of the bounding box """
        return self.bbox[2:]

    @property
    def scale(self):
        """ Larger side of the bounding box """
        return max(self.input_shape)

    def clear(self):
        """ Set the accumulator to 0 """
        self.A[:] = 0

    def transform(self, x):
        """
        Transform points x to the PCLines space and return polylines.

        Input
        -----
        x : ndarray
            Nx2 array with points

        Output
        ------
        p : ndarray
            Nx3 array with polyline coordinates for u, v, w parallel axes
        """
        _validate_points(x)
        x0,x1 = np.split(x,2,axis=1)
        return np.concatenate([self.norm_u(x1), self.norm_v(x0), self.norm_w(x1)], axis=1).astype("f")

    def inverse(self, l):
        """
        Transform a point from PCLines to homogeneous parameters of line
        """
        d = self.d
        m = self.scale - 1
        norm_v = Normalizer((0,d-1),(-m/2, m/2))

        u,v = l[:,1], l[:,0]
        u = u - (d - 1)
        v = norm_v(v)

        f = u < 0
        lines = np.array([f*(d+u)+(1-f)*(d-u), u, -v*d], "f").T  # TODO: add reference to eq in paper
        x,y = self.origin
        w,h = self.input_shape
        tx,ty = x+0.5*w, y+0.5*h
        lines[:,2] -= lines[:,0]*tx + lines[:,1]*ty
        return lines

    def valid_points(self, p):
        """ Check if the points fits the accumulator """
        return np.all(np.logical_and(p>=0, p<self.d), axis=1)

    def insert(self, x, weight=None):
        """ Insert observations x to the accumulator

        Inputs
        ------
        x : ndarray
            (N,2) observations
        weights : ndarray or None
            Weight of each observation in accumulator (defaults to 1 in None)

        Notes
        -----
        Any observation outside the bounding box is ignored.
        """
        p = self.transform(x)
        n = p.shape[0]

        if weight is None:
            weight = np.ones(n, np.float32)

        if weight.dtype != np.float32:
            weight = weight.astype(np.float32)

        valid = self.valid_points(p)
        p = p[valid]
        weight = weight[valid.flat]

        polys(p, weight, self.A)
        

    def find_peaks(self, t=0.8, prominence=2, prominence_radius=2, min_dist=1):
        """
        Retrieve locations with prominent local maxima in the accumulator
        """
        selem = disk(prominence_radius)
        p = dilation(self.A+1, selem=selem)/erosion(self.A+1, selem=selem)
        peaks = peak_local_max(self.A, threshold_rel=t, min_distance=min_dist, exclude_border=False)
        r,c = peaks[:,0], peaks[:,1]
        value = self.A[r,c]
        valid = p[r,c] > prominence
        return peaks[valid], value[valid]
