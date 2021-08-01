import os

import numpy as np

def normalize_and_save():

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

    write_dir = './ZS2020_tmp'
    fn = os.path.join(write_dir, 'X_ds.npy')

    with open(fn, 'rb') as f:
        X_ds = np.load(f)

    assert isinstance(X_ds, np.ndarray)
    print(X_ds.shape)

    X_ds_norm = np.array(
        list(
            map(
                lambda x: x / np.linalg.norm(x), 
                X_ds
            )
        )
    )

    fn = os.path.join(write_dir, 'X_ds_norm.npy')

    with open(fn, 'wb') as f:
        np.save(f, X_ds_norm)


if __name__ == '__main__':

    normalize_and_save()