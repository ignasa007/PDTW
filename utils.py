import numpy as np
from scipy.spatial import distance

# matlab_round: Round function as implemented in Matlab

def matlab_round(x):
    if x % 1 != 0.5:
        return round(x)
    else:
        return int(x+np.sign(x)*0.5)
vec_round = np.vectorize(matlab_round)


# Matlab like cosine distance calculation

def pdist2(X, Y, normalized=False):

    if type(X) != np.ndarray or type(Y) != np.ndarray:
        raise ValueError(f'Input arrays should be of type numpy.ndarray, but are of type {type(X)} and {type(Y)}.') 
    
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    if len(Y.shape) == 1:
        Y = np.expand_dims(Y, axis=0)

    if X.shape[1:] != Y.shape[1:]:
        raise ValueError(f"Dimensions of the input arrays don't match. X has constituent arrays of dimension {X.shape[1:]} and Y has constituent arrays of dimension {Y.shape[1:]}.")

    return np.array([
        [
            distance.cosine(X[i], Y[j]) if normalized else np.dot(X[i], Y[j])
            for j in range(Y.shape[0])
        ]
        for i in range(X.shape[0])
    ])


# dp2: DTW Implementation

def dp2(M):

    # M: Matrix with probabilities corresponding to the frame level distance between the segments
    # M.shape = (Frames in first segment, Frames in second segment)

    # r, c: Number of rows and columns in the input affinity matrix
    r, c = M.shape

    # D: 2D array holding costs 
    D = np.zeros(
        (r+1, c+1)
    )
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0
    D[1:r+1, 1:c+1] = M

    # phi: Trace-back
    phi = np.zeros(
        (r+1, c+1)
    )

    for i in range(1, r+1):
        for j in range(1, c+1):
            # Scale the 'longer' steps to discourage skipping ahead
            kk1, kk2 = 2, 1
            cost = D[i, j]
            opts = [
                D[i-1, j-1] + cost, 
                D[max(0, i-2), j-1] + cost*kk1, 
                D[i-1, max(0, j-2)] + cost*kk1, 
                D[i-1, j] + cost*kk2, 
                D[i, j-1] + cost*kk2
            ]
            dmax, trace_back = np.min(opts), np.argmin(opts)
            D[i, j], phi[i, j] = dmax, trace_back

    # Trace-back from top left
    i, j, p, q = r, c, [r-1], [c-1]
    while i>1 and j>1:
        trace_back = phi[i, j]
        if trace_back == 0:
            i -= 1
            j -= 1
        elif trace_back == 1:
            i -= 2
            j -= 1
        elif trace_back == 2:
            i -= 1
            j -= 2
        elif trace_back == 3:
            i -= 1
        elif trace_back == 4:
            j -= 1
        else:
            raise AssertionError(f'Trace back value outside expected range. Position {(i, j)}, trace_back = {trace_back}.')
        p.insert(0, i-1)
        q.insert(0, j-1)

    return p, q


# Matlab find function

def find(c):

    left, right = np.where(c == np.min(c))
    return left.item(), right.item()


# Used so that the code does not calculate all the pairwise distances before choosing the top few

def insert(dist, ind, val, idx):

    for i in range(len(dist)):
        if dist[i] is None or dist[i] == np.nan or dist[i] > val:
            dist = dist[:i] + [val] + dist[i:-1]
            ind = ind[:i] + [idx] + ind[i:-1]
            break

    return dist, ind