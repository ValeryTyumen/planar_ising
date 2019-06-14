import numpy as np


def repeat_int(value, count):

    array = np.zeros(count, dtype=np.int32)
    array[:] = value

    return array

def repeat_bool(value, count):

    array = np.zeros(count, dtype=np.bool_)
    array[:] = value

    return array

def repeat_float(value, count): 

    array = np.zeros(count, dtype=np.float32)
    array[:] = value

    return array
