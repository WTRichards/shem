import numpy as np
import shem.geometry

def superposition(N, function_dict, theta, phi, *args):
    '''
    Accepts the parameters:
        N - Int. The number of rays to sample from the distribution.
        theta - Float.
        phi - Float.
        function_dict - Dictionary of numbers indexed by functions defined in shem.source_functions.
    Returns:
        rays - An N x 3 NumPy array of direction vectors.
    '''
    rays = np.zeros([N, 3], dtype=np.dtype('float16'))

    functions = np.array(list(function_dict.keys()))
    abundance = np.array(list(function_dict.values()))
    rel_abundance = abundance / np.sum(abundance)
    choice_criteria = np.cumsum(rel_abundance)
    choice_criteria = np.insert(choice_criteria, 0, 0)
    
    # Choose each function with a weighting taken from rel_abundance.
    function_choice = np.random.rand(N)
    for r in range(N):
        for f in range(len(functions)):
            if choice_criteria[f] <= function_choice[r] and function_choice[r] < choice_criteria[f+1]:
                rays[r] = functions[f](theta, phi, args)
                break
    # Normalise the direction vectors.
    # Arguably the functions should do this but this is convenient.
    rays /= np.broadcast_to(np.linalg.norm(rays, axis=-1), (3,N)).T
    return rays

def delta(theta, phi, *args):
    '''
    Accepts the parameters:
        theta - Float.
        phi - Float.
    Returns:
        A length 3 NumPy array pointing in the (theta, phi) direction.
    '''
    a_polar = np.array([1, theta, phi])
    a = shem.geometry.polar2cart(a_polar, radians=True)
    return a

# Identical function for testing purposes.
def delta_(theta, phi, *args):
    return delta(theta, phi, *args)
