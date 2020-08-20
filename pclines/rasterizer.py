import numpy as np
import numba as nb

@nb.njit
def line(xr, yr, w, arr):
    x0,x1 = xr
    y0,y1 = yr
    dy = (y1-y0) / (x1-x0)
    for step,x in enumerate(range(x0, x1)):
        y = int(y0 + (dy * step))
        arr[y,x] += w

@nb.njit
def polys(p, weight, arr):
    n = weight.size
    d = arr.shape[0]
    for i in range(n):
        u,v,w = p[i]
        wt = weight[i]
        # rasterize (u,v), (0,d-1)
        line((0, d), (u, v), wt, arr)
        # rasterize (v,w), (d, 2d-1)
        line((d, 2*d-1), (v, w), wt, arr)
    

