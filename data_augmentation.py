import numpy as np


def data_augmentation(X, Y):
    map_dict = {'T': 'A', 'A': 'T', 'C': 'G', 'G': 'C'}
    out = []
    for string in X:
        out.append(''.join(x if x not in map_dict else map_dict[x] for x in string))
    return np.append(X, out), np.concatenate([Y, Y])
