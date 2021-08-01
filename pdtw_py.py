import os
import pickle 

import numpy as np

from load_data import loadData
from downsampled_segments import downsampledSegments
from candidate_matches import candidateMatches
from random_dist import frameLevelDist
from high_res_pdtw import highResMatching
                

def main(config):

    # First we make L frames segents form the entire speech corpus and then downsample it to M frame segments

    print('Getting downsampled segments.')

    # Number of segments of L frames is replaced by None in dimensions

    # X_ds: L frame segments downsampled to M frames and features concatenated,
    # X_ds.shape = (None, M*d)

    # X_ds_norm: X_ds normalized to reduce each M*d dimensional vector's length to 1
    # X_ds_norm.shape = (None, M*d)

    # X_ind: sequence of indices corresponding to entries of X (used later),
    # X_ind.shape = (None, L, 2)

    fn1 = os.path.join(write_dir, 'X_ds_norm.npy') if config['normalized'] else os.path.join(write_dir, 'X_ds.npy')
    fn2 = os.path.join(write_dir, 'X_ind.npy')

    if os.path.exists(fn1) and os.path.exists(fn2):
        with open(fn1, 'rb') as f1, open(fn2, 'rb') as f2:
            X_ds, X_ind = np.load(f1), np.load(f2)

    else:
        X_ds, X_ind = downsampledSegments(
            seqshift=config['seqshift'],
            seqlen=config['seqlen'],
            n_slices=config['n_slices'],
            normalized=config['normalized']
        )
        with open(fn1, 'wb') as f1, open(fn2, 'wb') as f2:
            np.save(f1, X_ds), np.save(f2, X_ind)

    # Find nearest segments to each segment using probabilistic distance

    print('Matching downsampled segments.')

    # I: Stores candidate indices
    # I.shape = (None, nearest_to_check)

    fn = os.path.join(write_dir, 'I.npy')

    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            I = np.load(f)

    else:
        I = candidateMatches(
            X_ds=X_ds, 
            nearest_to_check=config['nearest_to_check'],
            seqlen=config['seqlen'],
            seqshift=config['seqshift'],
            expansion=config['expansion'],
            alpha=config['alpha'],
            comps_GMM=config['comps_GMM'],
            write_dir=write_dir,
            normalized=config['normalized']
        )
        with open(fn, 'wb') as f:
            np.save(f, I)

    # X_ds not used anymore so delete it

    del X_ds

    # High-resolution alignment with Probabilistic DTW

    print('Calculating chance-level for high resolution alignments using GMM.')

    # Calculate statistics on pairwise frame-level distances

    # mu, sigma, weight: Parameters of the GMM fitted to random sample of frame level distances 

    fn = os.path.join(write_dir, 'GMM_params.pickle')

    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            mu, sigma, weight = pickle.load(f)

    else:
        mu, sigma, weight = frameLevelDist(
            mat_fn=config['mat_fn'], 
            X_ind=X_ind,
            expansion=config['expansion'],
            comps_GMM=config['comps_GMM'],
        )
        with open(fn, 'wb') as f:
            pickle.dump((mu, sigma, weight), f)

    # Calculate real alignment paths for the candidates from Stage 1

    print('Aligning well-matching segments with a higher resolution.')

    # patterns: Dictionary of arrays of equal length, 
    # For each match, 
    #   'onset' holds value of start frame of both segments in their respective utterance
    #   'offset' holds value of end frame of both segments in their respective utterance
    #   'signal' holds value of utterance index for both segments
    #   'dist' holds value of average cosine distance along the aligned path

    patterns, count = highResMatching(
        X_ind=X_ind, 
        I=I,
        m=mu,
        v=sigma,
        w=weight,
        alpha=config['alpha'], 
        expansion=config['expansion'], 
        comps_GMM=config['comps_GMM'],
        duration_thr=config['duration_thr']
    )

    for key in patterns:
        patterns[key] = patterns[key][:count, :]

    with open(os.path.join(write_dir, 'patterns.pickle'), 'wb') as f:
        pickle.dump(patterns, f)


if __name__ == '__main__':

    os.chdir(
        os.path.join(
            os.path.dirname(
                os.path.realpath(
                    __file__
                )
            ), # Current file's directory
            os.pardir, 
            'PDTW'
        )
    )

    mat_fn = './speechdb/feats/MFCC_features_train_english.mat'
    write_dir = './ZS2020_tmp'

    # F: List containing MFCC features of training utterance n from {1... N},
    # Each element is an array of shape (Number of frames in the specific utterance, d)
    F = loadData(mat_fn)

    config = {
        'alpha': 1e-3,
        'process_id': 'default',
        'seqlen': 20,
        'seqshift': 10,
        'nearest_to_check': 5,
        'duration_thr': 5,
        'n_slices': 4,
        'expansion': 25,
        'comps_GMM': 1,
        'normalized': True
    }

    main(
        config=config,
    )


'''

is_nan = np.sum(np.isnan(X_ds), axis=1) != 0

D_r = []
D = np.empty((X_ds.shape[0], config['nearest_to_check'])).fill(np.nan)
I = np.empty((X_ds.shape[0], config['nearest_to_check'])).fill(np.nan)

olap_frames = np.ceil(
    config['seqlen'] / config['seqshift'] / 2 +
    config['expansion'] / config['seqlen'] +
    5
)

for i in tqdm(range(X_ds.shape[0])):

    all_dist, dist_near = [], []

    for j in range(X_ds.shape[0]):

        if is_nan[i] or is_nan[j]:
            dist_near.append(np.nan)
        else:
            cos_dist = distance.cosine(X_ds[i, :], X_ds[j, :])
            all_dist.append(cos_dist)
            if abs(i-j) <=  olap_frames:
                dist_near.append(np.nan)
            else:
                dist_near.append(cos_dist)

    D_r.extend(random.sample(all_dist, config['nearest_to_check']))
    
    ind_near = np.argsort(dist_near)[:config['nearest_to_check']]
    dist_near = np.sort(dist_near)[:config['nearest_to_check']]

    ind_near[np.isnan(dist_near)] = np.nan

    D[i, :] = dist_near
    I[i, :] = ind_near

D_r = np.array(D_r)[~np.isnan(D_r)]
m, v, w = eng.gaussmix(D_r, [], [], comps_GMM)
i = np.argsort(m)
m, v, w = map(lambda arr: arr[i], (m, v, w))

for i in tqdm(range(X_ds.shape[0])):

    if comps_GMM == 2:
        data_prob = norm.cdf(D[i, :], m[0], np.sqrt(v[0])) * w[0] + norm.cdf(D[i, :], m[1], np.sqrt(v[1])) * w[1]
    elif comps_GMM == 1:
        data_prob = norm.cdf(D[i, :], m[0], np.sqrt(v[0]))
    else:
        raise ValueError(f'Wrong number of GMM components. Expected 1/2, received {comps_GMM}')

    D[i, data_prob > config['alpha']] = np.nan
    I[i, data_prob > config['alpha']] = np.nan

'''