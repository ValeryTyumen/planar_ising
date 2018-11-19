import numpy as np
from numba import jit
from numba.types import int32, boolean, float32
import os


@jit(int32[:](int32, int32), nopython=True)
def repeat_int(value, count):

    array = np.zeros(count, dtype=np.int32)
    array[:] = value

    return array

@jit(boolean[:](boolean, int32), nopython=True)
def repeat_bool(value, count):

    array = np.zeros(count, dtype=np.bool_)
    array[:] = value

    return array

@jit(float32[:](float32, int32), nopython=True)
def repeat_float(value, count): 

    array = np.zeros(count, dtype=np.float32)
    array[:] = value

    return array

def get_numba_type(class_):

    numba_disable_jit_variable = 'NUMBA_DISABLE_JIT'

    if numba_disable_jit_variable not in os.environ or \
            os.environ[numba_disable_jit_variable] != '1':
        return class_.class_type.instance_type

    return int32
