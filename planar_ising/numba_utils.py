import os
from numba.types import int32


def get_numba_type(class_):

    numba_disable_jit_variable = 'NUMBA_DISABLE_JIT'

    if numba_disable_jit_variable not in os.environ or \
            os.environ[numba_disable_jit_variable] != '1':
        return class_.class_type.instance_type

    return int32
