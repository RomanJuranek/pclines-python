import numpy as np


def line_segments_from_homogeneous(lines, bbox):
    x,y,w,h = bbox
    
    # Corner points
    A = np.array([x,y,1])
    B = np.array([x+w,y,1])
    C = np.array([x+w,y+h,1])
    D = np.array([x,y+h,1])

    # Cross product of pairs of corner points
    edges = [
        np.cross(a,b) for a,b in [[A,B],[B,C],[C,D],[D,A]]
    ]

    # Cross product of line params with edges
    intersections = [
        np.cross(lines, e) for e in edges
    ]

    # Normalize
    normalized = [
        p[:,:2] / p[:,-1].reshape(-1,1) for p in intersections
    ]

    X = []
    Y = []
    
    for p in zip(*normalized):
        P = []
        for (u,v) in p:
            if (x <= u <= x+w) and (y <= v <= y+h):
                P.append( (u,v) )
        if len(P) == 2:
            (x0,y0), (x1,y1) = P
            X.append( (x0,x1) )
            Y.append( (y0,y1) )
        else:
            X.append(None)
            Y.append(None)

    return X, Y