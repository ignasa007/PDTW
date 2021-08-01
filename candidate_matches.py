import os
import random
import copy
from tqdm import tqdm

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from utils import pdist2, insert


# distGMM: Returns attributes of the GMM fitted to cosine distances between randomly selected pairs of segments 

def distGMM(X_ds_tmp, nearest_to_check, comps_GMM, write_dir, normalized=False):

    # Measure distances between random pairs of points

    print('Calculating random distances.')

    # D_r: List of random data points as cosine distances between two segments

    fn = os.path.join(write_dir, 'D_r.npy')

    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            D_r = np.load(f)

    else:
        D_r = []
        for i in tqdm(range(X_ds_tmp.shape[0])):
            sample = random.sample(list(range(X_ds_tmp.shape[0])), nearest_to_check)
            tmp = pdist2(
                X_ds_tmp[i, :], 
                X_ds_tmp[sample, :],
                normalized=normalized
            ).ravel().tolist()
            D_r.extend(tmp)
        D_r = np.array(D_r)[~np.isnan(D_r)]
        with open(fn, 'wb') as f:
            np.save(f, D_r)


    # Fitting a Gaussian miture model to the random distances data.
    
    GMM = GaussianMixture(
        n_components=comps_GMM,
        covariance_type='diag',
        tol=0.0001,
        max_iter=1000,
        init_params='kmeans',
        verbose=1
    ).fit(D_r.reshape(-1, 1))

    m, v, w = GMM.means_, GMM.covariances_, GMM.weights_
    m, v, w = map(np.ravel, (m, v, w))
    i = np.argsort(m)
    m, v, w = map(lambda arr: arr[i], (m, v, w))

    return m, v, w


# candidateMatches: Calculates the probability of observing calculated distances between all pairs of segments and
# returns an array of indices of segments closest to each segment and another array with corresponding distances  

def candidateMatches(X_ds, nearest_to_check, seqlen, seqshift, expansion, alpha, comps_GMM, write_dir, normalized=False):

    # Make a copy without NaNs to measure distance distributions

    X_ds_tmp = copy.deepcopy(X_ds)
    X_ds_tmp = X_ds_tmp[np.sum(np.isnan(X_ds_tmp), axis=1) == 0, :]

    # Modelling distances with a GMM

    print('Modeling random distance distribution with a GMM.')

    # m: Mixture means, v: Mixture variances, w: Mixture weights
    # m.shape = v.shape = w.shape = (comps_GMM, 1)
    # m.shape and v.shape have their second component 1 because we are modeling the distribtuion for one variable only

    means, variances, weights = distGMM(
        X_ds_tmp=X_ds_tmp, 
        nearest_to_check=nearest_to_check,
        comps_GMM=comps_GMM,
        write_dir=write_dir,
        normalized=normalized
    )

    # Boolean array indicating if a segment has any NaN values
    # Needed because scipy.spatial.distance.cosine returns 0 if there's NaN in either of the input arrays

    is_nan = np.sum(np.isnan(X_ds), axis=1) != 0

    # D: Stores distance probabilities
    # I: Stores candidate indices
    # D.shape = I.shape = (None, nearest_to_check)

    D = np.empty(
        (X_ds.shape[0], nearest_to_check)
    ).fill(np.nan)

    I = np.empty(
        (X_ds.shape[0], nearest_to_check)
    ).fill(np.nan)

    # olap_frames: Number of frames in the neighborhood to ignore for similarity calculation

    olap_frames = np.ceil(
        seqlen / seqshift / 2 +
        expansion / seqlen +
        5
    )

    for i in tqdm(range(X_ds.shape[0])):

        # dist_near: List of cosine distances of a segment from all other segment

        dist_near = [None for _ in range(nearest_to_check)]
        ind_near = [None for _ in range(nearest_to_check)]

        for j in range(i+1, X_ds.shape[0]):

            if is_nan[i] or is_nan[j] or abs(i-j) <= olap_frames:
                try:
                    idx = dist_near.index(None)
                    dist_near[idx], ind_near[idx] = np.nan, np.nan
                except ValueError:
                    continue
            else:
                dist_near, ind_near = insert(
                    dist_near,
                    ind_near,
                    pdist2(
                        X_ds[i, :], 
                        X_ds[j, :],
                        normalized=normalized
                    ).item(),
                    j
                )

        # Fitting these distances to the GMM calculated before

        if comps_GMM == 2:
            data_prob = norm.cdf(dist_near, means[0], np.sqrt(variances[0])) * weights[0] + \
                        norm.cdf(dist_near, means[1], np.sqrt(variances[1])) * weights[1]
        elif comps_GMM == 1:
            data_prob = norm.cdf(dist_near, means[0], np.sqrt(variances[0]))
        else:
            raise ValueError(
                f'Wrong number of GMM components. Expected 1/2, received {comps_GMM}'
            )

        # Remove those segments which are too far

        dist_near[data_prob > alpha] = np.nan
        ind_near[data_prob > alpha] = np.nan

        # Set the distances and the indices in the main arrays - D and I

        D[i, :] = dist_near
        I[i, :] = ind_near

    # Make sure that the same pair does not occur twice in reverse order

    # for i in range(I.shape[0]):
    #     for j in range(I.shape[1]):
    #         tmp = I[i, j]
    #         if not np.isnan(tmp):
    #             I[tmp, I[tmp, :] == i] = np.nan

    np.save(os.path.join(write_dir, 'D.npy'), D)
    np.save(os.path.join(write_dir, 'I.npy'), I)

    assert (D==None) == (I==None)
    
    D = np.where(D==None, np.nan, D)
    I = np.where(I==None, np.nan, I)

    # Save D and I

    return I


if __name__ == '__main__':

    write_dir = './ZS2020_tmp'