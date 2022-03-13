try:
    import cupy
except:
    pass

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
    sortarr     = array[cupy.lexsort(array.T[::-1])]
    mask        = cupy.empty(array.shape[0], dtype=cupy.bool_)
    mask[0]     = True
    mask[1:]    = cupy.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

# Coordinates
R, THETA, PHI = 0, 1, 2
X, Y, Z = 0, 1, 2

# Small number. Can't be too small since we need to multiply vectors by it and their components might round to 0...
DELTA = 10**-6

