"""
PCLines transform for line detection

This package implements the method from
    Dubska et al, PCLines - Line detection with parallel coordinates, CVPR 2011

"""

import numpy as np
import numba as nb
from skimage.feature import peak_local_max
from skimage.morphology.grey import erosion, dilation


def linear_transform(src, dst):
    (s0, s1), (d0,d1) = src, dst
    w = (d1 - d0) / (s1 - s0)
    b = d0 - w*s0
    return w, b


class Normalizer:
    def __init__(self, src_range, dst_range=(0,1)):
        self.w, self.b = linear_transform(src_range, dst_range)
        self.wi, self.bi = linear_transform(dst_range, src_range)

    def transform(self, x):
        """ src -> dst """
        return self.w * x + self.b

    def inverse(self, x):
        """ dst -> src """
        return self.wi * x + self.bi

    __call__ = transform


def accumulate(x, bbox=None, d=256):
    """
    Accumulate observation in PCLines space

    bbox : tuple
        (x,y,w,h) format
    """
    # Create accumulator
    acc_shape = d,2*d-1
    A = np.zeros(acc_shape, "f")  # The accumulator

    if bbox is None:
        # autodetection of bbox
        pass

    # Normalizers
    def normalizers():
        x,y,w,h = bbox
        shape = (w,h)
        ranges = d * np.array(shape)/max(shape)
        ofs = (d-ranges) / 2
        (x0,y0),(x1,y1) = ofs, (ranges-1)+ofs
        print(bbox)
        norm0 = Normalizer((y,y+h), (y1, y0))
        norm1 = Normalizer((x,x+w), (x0, x1))
        norm2 = Normalizer((y,y+h), (y0, y1))
        return norm0, norm1, norm2

    norm0, norm1, norm2 = normalizers()

    x = [
        norm0(x[:,1]),
        norm1(x[:,0]),
        norm2(x[:,1])
    ]

    for a,b,c in zip(*x):  # remove space wraping
        t_part = np.linspace(a,b,d)
        s_part = np.linspace(b,c,d)
        c = np.arange(2*d-1,dtype="i")
        r = np.concatenate([t_part, s_part[1:]]).astype("i")
        #print(r,c)
        A[r,c] += 1

    return A


def lines_parameters(peaks, d, bbox):
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

    l = np.array([f*(d+u)+(1-f)*(d-u), u, -v*d], "f").T

    tx,ty = x+0.5*w, y+0.5*h

    l[:,2] -= l[:,0]*tx + l[:,1]*ty 
    
    return l
    


# @nb.njit("(f4[:,:],f4[:,:])")
# def rasterize_lines(lines, acc):
#     """
#     """
#     d = acc.shape[1]
#     for i in range(lines.shape[0]):
#         y0, y1 = lines[i]
#         dy = (y1 - y0) / (d-1)
#         y = y0
#         for x in range(d):
#             acc[x,y] += 1
#             y += dy



def find_peaks(A, t):
    prominence = dilation(A+1)/erosion(A+1)
    peaks = peak_local_max(A, threshold_abs=t, min_distance=1)
    r,c = peaks[:,0], peaks[:,1]
    value = A[r,c]
    valid = prominence[r,c] > 1.3
    return peaks[valid], value[valid]
    