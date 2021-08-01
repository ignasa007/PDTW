from tqdm import tqdm

import numpy as np
from statistics import mode
from scipy.stats import norm

from utils import pdist2, dp2, find


def highResMatching(X_ind, I, m, v, w, alpha, expansion, comps_GMM, duration_thr):

    chance_probs = np.array([alpha**k for k in range(1, 1001)])
    patterns = {
        'onset': np.zeros(
            (int(1e7), 2)
        ),
        'offset': np.zeros(
            (int(1e7), 2)
        ),
        'signal': np.zeros(
            (int(1e7), 2)
        ),
        'dist': np.zeros(
            (int(1e7), 1)
        ),
        'id': np.zeros(
            (int(1e7), 2)
        )
    }
    pat_count, id_count = 0, 0

    # Iterate over segments in the data
    for k1 in tqdm(range(X_ind.shape[0])):

        # First segment in the pair
        a1, b1 = X_ind[k1, :, 0], X_ind[k1, :, 1]
        signal_1 = mode(a1)
        b1 = b1[a1 != signal_1]
        b1 = np.array(
            range(
                max(0, min(b1)-expansion),
                min(max(b1)+expansion+1, F[signal_1].shape[0])
            )
        )
        Y1 = F[signal_1][b1, :]

        # Iterate over all of its nearest segments discovered earlier
        for k2 in I[k1, :]:

            # Second segment in the pair
            a2, b2 = X_ind[k2, :, 0], X_ind[k2, :, 1]
            signal_2 = mode(a2)
            b2 = b2[a2 != signal_2]
            b2 = np.array(
                range(
                    max(0, min(b2)-expansion),
                    min(max(b2)+expansion+1, F[signal_2].shape[0])
                )
            )
            Y2 = F[signal_2][b2, :]
            
            # Affintiy matrix between the two segments calculated as frame level cosine distance
            M = pdist2(Y1, Y2)

            # Map M with GMM fitted earlier on random sample of distances form the entire corpus
            if comps_GMM == 2:
                M = norm.cdf(M, m[0], np.sqrt(v[0])) * w[0] + norm.cdf(M, m[1], np.sqrt(v[1])) * w[1]
            elif comps_GMM == 1:
                M = norm.cdf(M, m[0], np.sqrt(v[0]))
            else:
                raise ValueError(f'Wrong number of GMM components. Expected 1/2, received {comps_GMM}')

            # Pairs of frames in the two segments corresponding to the aligned path
            p, q = dp2(M)

            # Probability values from the GMM over the aligned path 
            short_path = np.zeros(len(p), 1)
            for t, (i, j) in enumerate(zip(p, q)):
                short_path[t] = M[i, j]
            short_path_log = np.log(short_path)
            
            # Holds the log likelihood for all sub-paths of the aligned path
            c = np.zeros(
                (len(short_path), len(short_path))
            )
            for left_bound in range(len(short_path)):
                for right_bound in range(left_bound+1, len(short_path)):
                    prob_path = sum(short_path_log[left_bound:right_bound+1])
                    # Summing over the log probabilities and penalizing for short sub-paths 
                    c[left_bound, right_bound] = prob_path - chance_probs[right_bound-left_bound]

            # Left bound and right bound for the sub-path with lowest log likelihood over the aligned path
            left_bound, right_bound = find(c)

            # If path is long enough, save
            if right_bound-left_bound+1 >= duration_thr:

                onset_1, end_1 = b1[p[left_bound]], b1[p[right_bound]]
                onset_2, end_2 = b2[q[left_bound]], b2[q[right_bound]]

                patterns['onset'][pat_count, 1] = onset_1
                patterns['offset'][pat_count, 1] = end_1
                patterns['signal'][pat_count, 1] = signal_1
                patterns['id'][pat_count, 1] = id_count

                patterns['onset'][pat_count, 2] = onset_2
                patterns['offset'][pat_count, 2] = end_2
                patterns['signal'][pat_count, 2] = signal_2
                patterns['id'][pat_count, 2] = id_count+1

                patterns['dist'][pat_count, 1] = np.mean(short_path[left_bound:right_bound+1])

                pat_count += 1
                id_count += 2

    return patterns, pat_count


if __name__ == '__main__':

    from load_data import loadData
    
    F = loadData()