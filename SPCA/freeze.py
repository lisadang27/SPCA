import inspect
from functools import partial

import SPCA
from SPCA import *

# FIX: Add a docstring for this function or remove it
def get_lambdaparams(function):
    return inspect.getfullargspec(function).args[1:]

# FIX: Add a docstring for this function
def get_fitted_params(function, dparams):
    if type(function) == partial:
        name = function.func.__name__
    else:
        name = function.__name__
    
    if name=='detec_model_bliss':
        if 'sigF' in dparams:
            params = []
        else:
            params = ['sigF']
    else:
        params = get_lambdaparams(function)
        params = [param for param in params if param not in dparams]
    return params

# FIX - this is currently empty!!!
def load_past_params(path):
    """Load the fitted parameters from a previous run.

    Args:
        path (string): Path to the file containing past mcmc result (must be a table saved as .npy file).

    Returns:
        ndarray: p0 (the previously fitted values)
    
    """
    
    return

def make_lambdafunc(function, dparams=[], obj=[], debug=False):
    """Create a lambda function called dynamic_funk that will fix the parameters listed in dparams with the values in obj.

    Note: The module where the original function is needs to be loaded in this file.
    
    Args:
        function (string): Name of the original function.
        dparams (list, optional): List of all input parameters the user does not wish to fit (default is None.)
        obj (string, optional): Object containing all initial and fixed parameter values (default is None.)
        debug (bool, optional): If true, will print mystr so the user can read the command because executing it (default is False).

    Returns:
        function: dynamic_funk (the lambda function with fixed parameters.)
    
    """
    
    full_args  = inspect.getfullargspec(function).args[1:]
    freeze_kwargs = dict([[dparams[i], obj[dparams[i]]] for i in range(len(dparams)) if dparams[i] in full_args])
    dynamic_funk = partial(function, **freeze_kwargs)
    
    if debug:
        print(inspect.getfullargspec(dynamic_funk).args[1:])
        print()
    
    return dynamic_funk