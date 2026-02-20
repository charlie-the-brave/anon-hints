import pickle
import numpy as np
from copy import deepcopy
from common.extra import print_message


SENTINEL = -float('inf') # do not set to nan


def try_load_pkl(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print_message(f"error loading {path}: {e}", 1)
        return []

def mean_ci95(array, dim=0):
    if array.size == 0: return np.zeros(1,), np.zeros(1,), 0
    n_filtered = np.count_nonzero(array != SENTINEL, axis=dim)
    mean = np.mean(array, axis=dim, where=array != SENTINEL)
    std = np.std(array, axis=dim, where=array != SENTINEL)
    if len(mean.shape) == 1: mean = mean.reshape(-1, 1)
    if len(std.shape) == 1: std = std.reshape(-1, 1)
    return mean, 1.96 * std / np.sqrt(n_filtered), n_filtered.min()

def dim_sum(array, dim=0):
    return np.sum(array, axis=0, where=(array != SENTINEL), keepdims=True)

def dim_max(array, dim=0):
    return np.max(array, axis=0, where=(array != SENTINEL), keepdims=True)

def list_shape(ll):
    # assumption: last dimension has 1-d elements (no zero-d)
    # compute shape as extent of each dimension in nested lists
    return max(list(map(list_shape,ll)), default=[]) + [len(ll)] if isinstance(ll,list) or isinstance(ll,np.ndarray) else [1]

def to_matrix(lists):
    try:
        return np.array(lists)
    except:
        shape = list_shape(lists)
        shape.reverse()
        assert shape[-1] == 1, "last dimension must be 1"

        # TODO: simpler implementation: create array of sentinels with shape, fill with list values
        # fill missing values in list with sentinel (slow)
        # 0) detect missing values along dimension,
        # 1) create new dimension for missing values,
        # 2) next iteration fills the new dimension to match shape
        dim, ref_list = 0, lists[0]
        lists_filled = deepcopy(lists)
        stak = [lists_filled]
        while len(stak) > 0:
            # process dimension in stack
            for _ in range(len(stak)):
                d = stak.pop(0)
                if (shape[dim] - len(d)) > 0:
                    if isinstance(ref_list, list) or isinstance(ref_list, np.ndarray):
                        # create new dimension to fill in next iteration
                        d.extend((shape[dim] - len(d))*([[SENTINEL]]))
                    else:
                        # don't create new dimension for last dim
                        d.extend((shape[dim] - len(d))*([SENTINEL]))

                # recurse inner dimensions
                stak.extend([inner for inner in d if isinstance(inner, list)])
            dim += 1
            ref_list = ref_list[0] if dim < len(shape) - 1 else ref_list

        return np.array(lists_filled)

