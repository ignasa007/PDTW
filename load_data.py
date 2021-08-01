import h5py
from tqdm import tqdm

import numpy as np


# loadData: Loads the list of MFCC features of each utterance from memory

def loadData(mat_fn):

    with h5py.File(mat_fn) as f:

        # references: array of references to the MFCC features

        print('Loading data.')

        references = f.get('F_train')[0]

        # F: List containing MFCC features of training utterance n from {1... N},
        # Each element is an array of shape (Number of frames in the specific utterance, d),
        # len(data) = Number of utterances

        print('Formatting data.')

        F = [np.array(f.get(reference)).T for reference in tqdm(references)]

    return F


# mfccFeatures: Returns MFCC features formatted as segments of L frames

def mfccFeatures():

    # total_len: sum of lengths of the time dimension of MFCC features over all the utterances

    total_len = sum([
        feats.shape[0] for feats in F
    ])

    # F_all: MFCC features for all frames of all utterances,
    # F_all.shape = (Total number of feature vectors in the corpus, d)

    F_all = np.zeros(
        (total_len, F[0].shape[1])
    )

    # F_ind: sequence of indices corresponding to entries of F_all,
    # if (x, y) = F_ind[i], then F_all[i] corresponds to the MFCC features of
    # y-th frame of x-th utterance,
    # F_ind.shape = (Total number of feature vectors in the corpus, 2)

    F_ind = np.zeros(
        (total_len, 2)
    )

    wloc = 0
    for i, feats in enumerate(F):
        length = feats.shape[0]
        F_all[wloc:wloc+length, :] = feats
        F_ind[wloc:wloc+length, 0] = i
        F_ind[wloc:wloc+length, 1] = list(range(length))
        wloc = wloc+length

    # Replace all infinities and nans with 0

    F_all = np.nan_to_num(
        F_all, nan=0.0, posinf=0.0, neginf=0.0
    )

    return F_all, F_ind


if __name__ == '__main__':

    mat_fn = r'..\PDTW\speechdb\feats\MFCC_features_train_english.mat'
    
    F = loadData(mat_fn=mat_fn)