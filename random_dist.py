import numpy as np
from statistics import mode
from sklearn.mixture import GaussianMixture

from load_data import loadData
from utils import pdist2


# Helper function to get random position of a frame from an utterance and then extend it by E frames on both sides

def expandedSegment(X_ind, expansion, k=None, neq=None, random=True):

    if (k is None) == (not random):
        raise ValueError(f'Either k needs to be None and random True or k should be a positive integer and random False. Received k = {k} and random = {random}.')

    if k is not None and (not isinstance(k, int) or k < 0 or k >= X_ind.shape[0]):
        raise ValueError(f'k needs to be either a non-negative integer less than X_ind.shape[0] or None. Received k = {k} where X_ind.shape = {X_ind.shape}.') 
    
    # k: Index of the random L-frame-segment to consider

    if k is None:
        if neq is None:
            k = random.randint(0, X_ind.shape[0]-1)
        else:
            k = neq
            while k == neq:
                k = random.randint(0, X_ind.shape[0]-1)
    
    # a: Utterance index of the frames in the randomly selected segment
    # b: Frame index in the utterance corresponding to each of the frames in the segment
    
    a, b = X_ind[k, :, 0], X_ind[k, :, 1]

    # signal: Mode of a - needed so that the frames are from the same utterance

    signal = mode(a)
    b = b[a != signal]

    # Extending the window of frames in the utterance given by signal by E frames

    b = np.array(
        range(
            max(0, min(b)-expansion),
            min(max(b)+expansion+1, F[signal].shape[0])
        )
    )

    Y = F[signal][b, :]

    return k, Y


# frameLevelDist: Returns parameters of GMM fitted to frame level distances used for high resolution PDTW

def frameLevelDist(X_ind, expansion, comps_GMM):

    # all_dist: Take ~2M random samples from frame level distances

    all_dist = np.zeros(int(2e6))

    wloc = 0
    while wloc < 2e6:
        k, Y1 = expandedSegment(X_ind, expansion)
        _, Y2 = expandedSegment(X_ind, expansion, neq=k)
        M = pdist2(Y1, Y2)
        diag = np.diag(M)
        all_dist[wloc:wloc+diag.size] = np.ravel(diag)
        wloc += diag.size

    # Model frame level distances with a GMM

    GMM = GaussianMixture(
        n_components=comps_GMM,
        covariance_type='diag',
        tol=0.0001,
        max_iter=1000,
        init_params='kmeans',
        verbose=1
    ).fit(all_dist)

    m, v, w = GMM.means_, GMM.covariances_, GMM.weights_
    m, v, w = map(np.ravel, (m, v, w))
    i = np.argsort(m)
    m, v, w = map(lambda arr: arr[i], (m, v, w))

    return m, v, w


if __name__ == '__main__':

    from load_data import loadData
    
    F = loadData()