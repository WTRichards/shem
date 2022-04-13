import numpy as np
import cupy  as cp

from shem.definitions import *

# Sample from a continuous probability distriution
def pdist_sample(pdist, n=1, bounds=np.array([[0,],[1,]]), n_points=np.array([1024,]), use_gpu=True, **kwargs):
    '''
    Quantises a continuous probability distribution pdist in any number of variables.
    The bounds on these variables are given by the lists v_min and v_max.
    The variable point specifies the number of points to sample along that dimension.
    Note that the number of points we need to check scales as 2**dimensions.
    We return n points from this distribution as a list, like np.where.
    Because this function discretises each dimension:
    DO NOT USE THIS FOR SHARPLY PEAKED PROBABILITY DISTRIBUTIONS WITHOUT SPECIFYING THE BOUNDS
    '''
    if use_gpu:
        xp = cp
    else:
        xp = np

    # Get the number of dimensions from the bounds.
    dims = bounds.shape[1]
    assert dims == n_points.shape[0]

    # Create a meshgrid from the points. It is sparse and uses 'ij' indexing.
    points = [ xp.linspace(bounds[0][d], bounds[1][d], n_points[d]) for d in range(dims) ]
    points_mesh = xp.meshgrid(*points, sparse=True, indexing='ij')

    # Pass the sampled points to the probability distribution. Broadcasting is automatic.
    probability = pdist(*points_mesh, **kwargs)

    # Normalise the distribution.
    probability /= probability.sum()

    # Select n points from the probability distribution.
    choices = xp.unravel_index(xp.random.choice(int(n_points.prod()), size=n, p=probability.reshape(-1)), n_points)

    # Return a list of each coordinate point chosen.
    return [points[dim][choice] for dim, choice in enumerate(choices)]
