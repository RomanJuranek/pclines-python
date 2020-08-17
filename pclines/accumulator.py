"""
PCLines transform for line detection

This package implements the method from
    Dubska et al, PCLines - Line detection with parallel coordinates, CVPR 2011


Module
------
The module provides a high-level function for line detection in image
and also low-level functions for
* accumulation of observations to PCLines space,
* point mapping from PCLines space to homogeneous lines
that can be used to construct a custom PCLines transform of user-defined
edge points.


See also
--------
* pclines.accumulate
* pclines.find_peaks
* pclines.line_parameters


References
----------
[1] Dubska et al, PCLines - Line detection with parallel coordinates, CVPR 2011


"""


import numpy as np
import numba as nb
from skimage.feature import peak_local_max
from skimage.morphology.grey import erosion, dilation


def _linear_transform(src, dst):
    """ Parameters of a linear transform from range specifications """
    (s0, s1), (d0,d1) = src, dst
    w = (d1 - d0) / (s1 - s0)
    b = d0 - w*s0
    return w, b


class Normalizer:
    """
    Range mapping
    """
    def __init__(self, src_range, dst_range=(0,1)):
        # TODO: check ranges
        self.w, self.b = _linear_transform(src_range, dst_range)
        self.wi, self.bi = _linear_transform(dst_range, src_range)

    def transform(self, x):
        """ src -> dst mapping of x """
        return self.w * x + self.b

    def inverse(self, x):
        """ dst -> src mapping of x """
        return self.wi * x + self.bi

    __call__ = transform


def accumulate(x, w=None, bbox=None, d=256):
    """
    Accumulate observation in PCLines space

    bbox : tuple
        (x,y,w,h) format
    """
    # TODO: Check inputs


    # Create accumulator
    acc_shape = d,2*d-1
    A = np.zeros(acc_shape, "f")  # The accumulator

    # Axis normalizers
    def normalizers():
        x,y,w,h = bbox
        shape = (w,h)
        ranges = d * np.array(shape)/max(shape)
        ofs = (d-ranges) / 2
        (x0,y0),(x1,y1) = ofs, (ranges-1)+ofs
        norm0 = Normalizer((y,y+h), (y1, y0))
        norm1 = Normalizer((x,x+w), (x0, x1))
        norm2 = Normalizer((y,y+h), (y0, y1))
        return norm0, norm1, norm2

    norm0, norm1, norm2 = normalizers()

    # Coordinates on parallel axes
    x0,x1 = np.split(x,2,axis=1)
    x = np.concatenate([norm0(x1), norm1(x0), norm2(x1)], axis=1)

    # Rasterize the lines
    for a,b,c in x:  # remove space wraping
        t_part = np.linspace(a,b,d)
        s_part = np.linspace(b,c,d)
        c = np.arange(2*d-1,dtype="i")
        r = np.concatenate([t_part, s_part[1:]]).astype("i")
        #print(r,c)
        A[r,c] += 1

    return A


def lines(peaks, bbox, d):
    """
    Get homogeneous line parameters from location in the accumulator
    """
    u = peaks[:,1]
    v = peaks[:,0]
    #centrovanie
    u = u - (d - 1)
    x,y,w,h = bbox
    shape = w,h
    m = max(shape) - 1
    normV = Normalizer((0,d-1),(-m/2, m/2))
    v = normV(v)
    f = u < 0
    l = np.array([f*(d+u)+(1-f)*(d-u), u, -v*d], "f").T  # TODO: add reference to eq in paper
    tx,ty = x+0.5*w, y+0.5*h
    l[:,2] -= l[:,0]*tx + l[:,1]*ty 
    return l
    


@nb.njit("(f4[:,:],f4[:,:],i4)")
def rasterize_polylines(lines, acc, d):
    """
    """


def find_peaks(A, t):
    """
    Retrieve locations with prominent local maxima in the accumulator
    """
    prominence = dilation(A+1)/erosion(A+1)
    peaks = peak_local_max(A, threshold_abs=t, min_distance=5)
    r,c = peaks[:,0], peaks[:,1]
    value = A[r,c]
    valid = prominence[r,c] > 1.5
    return peaks[valid], value[valid]
    

def get_lines(image):
    """
    PCLines transform of an image
    """
    # TODO: Get edges
    # TODO: Accumulate
    # TODO: Locate peaks
    # TODO: Transform peaks to line parameters
    pass