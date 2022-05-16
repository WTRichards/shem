import numpy as np
# Disable NumPy warnings - we avoid performing certain checks on the GPU for speed.
np.seterr(all="ignore")
import cupy as cp
import tables as tb
import os,sys

import trimesh
import tqdm
import math
import json
import time
import shutil

from importlib.machinery import SourceFileLoader
import hashlib
import pickle

import shem.database
from   shem.definitions import *
import shem.default_config
import shem.display
import shem.geometry
import shem.mesh
import shem.optimisation
import shem.ray
import shem.scattering
import shem.simulation
import shem.source

######################
# Configuration File #
######################

def create_default(path):
    '''
    Creates the default configuration file at the path supplied.
    '''
    default_config_path = shem.default_config.__file__
    return shutil.copy(default_config_path, path)

def load_config(args):
    '''
    Load the configuration file as a module
    '''
    config_file = os.path.join(args.work_dir, args.config)
    conf = SourceFileLoader("conf", config_file).load_module()
    return conf

def get_settings(args=None, settings=None, defaults=None):
    '''
    Gets the true value of settings from the arguments provided, applying default values
    '''
    # Begin the recursion
    if settings is None and defaults is None:
        # Load the settings and defaults.
        if args is None:
            conf = shem.default_config
        else:
            conf = load_config(args)
        settings = conf.settings
        defaults = shem.default_config.defaults
        get_settings(args, settings, defaults)

    # Combine the settings and defaults
    if type(settings) is dict and type(defaults) is dict:
        settings_key_set = set(settings.keys())
        defaults_key_set = set(defaults.keys())
        # All keys available are the union of the key sets of defaults and settings.
        keys = settings_key_set.union(defaults_key_set)
        # Assign the provided settings to key k if possible, otherwise use the default.
        return {k: get_settings(args, settings[k], defaults[k]) if k in settings_key_set else defaults[k] for k in keys}

    # Prioritise the values defined in settings
    return settings
        

#########################
# Optimisation Settings #
#########################

def template_iterate(settings, template=None, function=lambda s, b, acc: b, acc=0, acc_manip=lambda acc, res: acc+1, return_acc=False, return_res=True):
    '''
    Return a dictionary like template with the values replaced as a function of the default values, the bounds in the template.
    '''
    
    if template is None:
        template = settings["meta"]["solver"]["template"]
    
    # Iterate over the template dictionary
    if type(template) is dict and type(settings) is dict:
        res = {}
        for k in template.keys():
            acc, res[k] = template_iterate(settings[k], template[k], function, acc, acc_manip, True)

        if return_acc and return_res:
            return acc, res
        elif return_res:
            return res
        elif return_acc:
            return acc
        else: 
            return

    # Assign new values using the function and increment the accumulator.
    elif (type(template) is tuple and (type(settings) is int or type(settings) is float)) or type(template) is list:
        res = function(settings, template, acc)
        acc = acc_manip(acc, res)
        return acc, res
    
    # Something has gone wrong.
    else:
        raise ValueError()

def settings_iterate(settings, template=None, function=lambda s, b, acc: s, acc=0, acc_manip=lambda acc, res: acc+1, return_acc=False, return_res=True):
    '''
    Return a dictionary like settings with the values replaced as a function of the default values, the bounds in the template.
    '''
    if template is None:
        template = settings["meta"]["solver"]["template"]


    # Iterate over the template dictionary. We do this in a bit of a weird way to preserve the ordering.
    if type(template) is dict and type(settings) is dict:
        res = {}
        not_in_template = set(settings.keys()).difference(template.keys())
        for k in template.keys():
            acc, res[k] = settings_iterate(settings[k], template[k], function, acc, acc_manip, True)
        for k in not_in_template:
            res[k] = settings[k]

        if return_acc and return_res:
            return acc, res
        elif return_res:
            return res
        elif return_acc:
            return acc
        else: 
            return

    # Assign new values using the function and increment the accumulator.
    elif (type(template) is tuple and (type(settings) is int or type(settings) is float)) or type(template) is list:
        res = function(settings, template, acc)
        acc = acc_manip(acc, res)
        return acc, res
    
    # Something has gone wrong.
    else:
        raise ValueError()

def get_parameters(settings):
    '''
    Give the specified template, extract the default values as a from settings.
    '''
    setting_value = lambda s, b, acc: s
    return template_iterate(settings, function=setting_value)

def get_parameters_count(settings):
    '''
    Given the specified template, count the number of different variable parameters.
    '''
    return template_iterate(settings, return_acc=True, return_res=False)

def get_parameters_bounds(settings):
    '''
    Give the specified template, get the bounds as a list.
    '''
    # Append the bound in the template
    append_bound = lambda acc, res: acc + [res]
    return template_iterate(settings, acc=[], acc_manip=append_bound, return_acc=True, return_res=False)

def check_settings_in_bounds(settings):
    '''
    Check that the settings supplied fall within the bounds specified in template.
    If they all do, return True.
    If one doesn't, return a dictionary like template with the bounds replaced by booleans representing proper values.
    '''
    # A function which checks if the setting is satisfies the bounds in template.
    is_in_bounds     = lambda s, b, acc: True if (type(b) is tuple and s >= b[0] and s <= b[1]) or (type(b) is list and s in b) else False
    others_in_bounds = lambda acc, res: acc and res

    # Check if all the settings satisfies the bounds in template.
    all_in_bounds, which_in_bounds = template_iterate(settings, function=is_in_bounds, acc=True, acc_manip=others_in_bounds, return_acc=True)

    if not all_in_bounds:
        return which_in_bounds
    else:
        return True

def randomise_settings_in_bounds(settings):
    '''
    Return a copy of settings where the parameters specified in template have been set to random values in the specified bounds.
    '''
    # A function which generates a random value which satisfies the bounds in template. We use NumPy since we manage the seeding for it.
    random_in_bounds = lambda s, b, acc: np.random.uniform(*b) if type(b) is tuple else b[np.random.randint(len(b))]
    return settings_iterate(settings, function=random_in_bounds)

def set_setting_values(settings, v):
    '''
    Return a copy of settings where the parameters specified in a list have been altered.
    '''
    assign_value = lambda s, b, acc: v[acc]
    return settings_iterate(settings, function=assign_value)

def get_setting_values(settings):
    '''
    Return the list of parameters from settings specified in the template.
    '''
    setting_value = lambda s, b, acc: s
    append_value = lambda acc, res: acc + [res]
    return template_iterate(settings, function=setting_value, acc=[], acc_manip=append_value, return_acc=True, return_res=False)
