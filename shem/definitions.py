import pickle
import hashlib

import numpy as np
import cupy  as cp

from skimage.restoration import denoise_nl_means, estimate_sigma

import cv2

# Convert a dictionary into an object.
# Makes config files so much nicer.
# https://joelmccune.com/python-dictionary-as-object/
class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
    # Can recover the dictionary locally
    def items(self):
        return self.__dict__.items()

###################################
# replacement for numpy.unique with option axis=0
###################################
# https://stackoverflow.com/questions/58662085/is-there-a-cupy-version-supporting-axis-option-in-cupy-unique-function-any
def cupy_unique_axis0(array):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cp.lexsort(array.T[::-1])]
    mask        = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0]     = True
    mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]



# Coordinates
R, THETA, PHI = 0, 1, 2
X, Y, Z = 0, 1, 2

# Small number. Can't be too small since we need to multiply vectors by it and their components might round to 0...
DELTA = 1e-6

# Hash functions

# https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
# md5 checksum of the contents of fname
def hash_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# md5 hash of a serialised dictionary
def hash_obj(dictname):
    return hashlib.md5(pickle.dumps(dictname)).hexdigest()

def get_xp(args):
    if args.use_gpu:
        return cp
    else:
        return np


# https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
# Denoising function
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=None)
denoise = lambda data: denoise_nl_means(data/data.max(), h=1.15 * np.mean(estimate_sigma(data/data.max())), fast_mode=False, **patch_kw)
    

