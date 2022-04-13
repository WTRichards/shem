import os, sys

import numpy as np
import cupy as cp

import shem.geometry

from shem.definitions import *

from shem.geometry import vector_dot       as dot
from shem.geometry import vector_cross     as cross
from shem.geometry import vector_norm      as norm
from shem.geometry import vector_normalise as normalise

def parameterise(a, b, t):
    xp = cp.get_array_module(a, b, t)
    return a*xp.expand_dims(t, -1) + b

def intersects_plane_time(a, b, n, c):
    return dot(n, c - b) / dot(n, a)

def reflect(a, n):
    xp = cp.get_array_module(a, n)
    return a - 2*xp.expand_dims(dot(n, a), -1)*n

