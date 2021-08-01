import math
from tqdm import tqdm

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from load_data import mfccFeatures
from utils import vec_round


# filterSilence: Filter out downsampled segments which are likely to belong to the silent part of speech

def filterSilence(X_ds, energy_dims):

    # E: Sum of energy components of the L frames in each segment,
    # E.shape = (None)

    E = np.sum(
        X_ds[:, energy_dims], 
        axis=1
    )

    # We fit the energies to a GMM to separate out 2 types of segments - silent and voiced

    print('Fitting GMM to energy levels to filter out silent segments.')

    data = E[~np.isnan(E)].reshape(-1, 1)

    GMM = GaussianMixture(
        n_components=2,
        covariance_type='diag',
        tol=0.0001,
        max_iter=1000,
        init_params='kmeans',
        verbose=1
    ).fit(data)

    # Ensuring that both components have considerable representation in the data 

    while min(GMM.weights_) < 0.05:
        GMM = GMM.fit(data)

    # m: Mixture means, v: Mixture variances, w: Mixture weights
    # m.shape = (2), v.shape = (2), w.shape = (2)

    m, v, w = GMM.means_, GMM.covariances_, GMM.weights_
    m, v, w = map(np.ravel, (m, v, w))

    # d: Distances of the segments from the mixture means
    # d.shape = (None, 2)

    print("Calculating distance of each of the segments' total energy level from the mixture means.")

    d = np.array([
        [
            abs(E[i]-m[j]) 
            for j in range(m.shape[0])
        ] 
        for i in tqdm(range(E.shape[0]))
    ])

    # idx: Index of the mixture to which a segment belongs - 0/1
    # idx.shape = (None)

    # counts: Number of points in each segment
    # counts.shape = (2)

    # i = Index of Mixture with higher number of segments - assumed to be the voiced cluster

    idx = np.argmin(d, axis=1)

    counts = np.array([
        np.sum(idx == 0), 
        np.sum(idx == 1)
    ])

    i = np.argmax(counts)

    # prob_sil: Probability that a segment is silent, calculated using voiced mixture
    # Set silent segments to NaN if p(silence) > thr

    prob_sil = 1 - norm.cdf(
        E, 
        loc=m[i], 
        scale=np.sqrt(v[i])
    ) * w[i]

    X_ds[prob_sil > 0.99, :] = np.nan

    return X_ds


# Get downsampled segments of M frames from the original data with the silenced frames removed

def downsampledSegments(seqshift, seqlen, n_slices, normalized=False):

    # F_all: MFCC features for all utterances,
    # F_all.shape = (Total number of feature vectors in the corpus, d)

    # F_ind: sequence of indices corresponding to entries of F_all,
    # if (x, y) = F_ind[i], then F_all[i] corresponds to the MFCC features of
    # y-th frame of x-th utterance,
    # F_ind.shape = (Total number of feature vectors in the corpus, 2)

    F_all, F_ind = mfccFeatures()

    # X: sequence of MFCC features of fixed-length segments of L frames,
    # X.shape = (None, L, d)

    X = np.zeros(
        (
            math.floor(
                1 + (F_all.shape[0] - seqlen) / seqshift
            ),
            seqlen, 
            F_all.shape[1]
        )
    )

    # X_ind: sequence of indices corresponding to entries of X
    # if (x, y) = X_ind[i, j], then X[i, j, :], the features of the j-th frame of i-th L-frame-segment,
    # correspond to the MFCC features of y-th frame of x-th utterance in data,
    # X_ind.shape = (None, L, 2)

    X_ind = np.zeros(
        (
            math.floor(
                1 + (F_all.shape[0] - seqlen) / seqshift
            ),
            seqlen, 
            2
        )
    )

    wloc, k = 0, 0

    while wloc <= F_all.shape[0]-seqlen:
        X[k, :, :] = F_all[wloc:wloc+seqlen, :]
        X_ind[k, :, :] = F_ind[wloc:wloc+seqlen, :]
        k = k+1
        wloc = wloc+seqshift

    # feat_dim: mfcc features size, default = 39

    feat_dim = X.shape[2]

    # bounds: Slicing indices for L frames in each segment to downsample to M frames
    # We divide the sequence of L frames into M windows with approximately equal frames in each window
    # Then we take average over these windows to get the downsampled segment

    bounds = vec_round(
        np.linspace(0, seqlen, num=n_slices+1)
    )

    # X_ds: L frame segments downsampled to M frames and features concatenated
    # X_ds.shape = (None, M*d)

    X_ds = np.zeros((X.shape[0], n_slices*feat_dim))

    for s in range(n_slices):
        X_ds[:, s*feat_dim:(s+1)*feat_dim] = np.mean(
            X[:, bounds[s]:bounds[s+1], :], 
            axis=1
        )

    # sildims: Indices of energy dimensions in X_ds

    energy_dims = range(0, X_ds.shape[1], F_all.shape[1])

    # Filter out silent segments

    X_ds = filterSilence(X_ds, energy_dims)

    # X_ds only used for cosine distance calculation, normalize to speed up calculation

    if normalized:
        X_ds = np.array(
            list(
                map(
                    lambda x: x / np.linalg.norm(x), 
                    X_ds
                )
            )
        )

    return X_ds, X_ind 