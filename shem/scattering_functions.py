import numpy as np
import shem.geometry

def superposition(function_dict, L, N, R, V):
    '''
    Accepts the parameters:
        N - Int. The number of rays to sample from the distribution.
        theta - Float.
        phi - Float.
        function_dict - Dictionary of numbers indexed by functions defined in shem.source_functions.
    Returns:
        rays - An N x 3 NumPy array of direction vectors.
    '''
    
    pixel = 0.0
    # Determine the strengths of each scattering distribution.
    strengths = np.array([d['strength'] for d in list(function_dict.values())])
    
    for _, func in enumerate(function_dict):
        # Use each source function to generate the required number of rays.
        # Concatenate the dictionaries and pass as **kwargs
        pixel += function_dict[func]['strength'] * func({**{'L': L, 'N': N, 'R': R, 'V': V}, **function_dict[func]})
    
    
    # Normalise the result
    pixel /= np.sum([d['strength'] for d in list(function_dict.values())])

    return pixel

def specular(kwargs):
    R = kwargs['R']
    V = kwargs['V']
    shininess = kwargs['shininess']
    # Can have multiple reflected vectors R but only a single viewer/detector V.
    # Shininess parametrisation from Blinn-Phong shading.
    return ((R*V).sum(-1)**shininess).sum()

def diffuse(kwargs):
    L = kwargs['L']
    N = kwargs['N']
    # Source vector L dot product with normal N.
    return ((L*N).sum(-1)).sum()

