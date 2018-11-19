import numpy as np
from enum import IntEnum


class SeparationClass(IntEnum):

    FIRST_PART = 0
    SECOND_PART = 1
    SEPARATOR = 2
    UNDEFINED = 3


UNDEFINED = np.int32(SeparationClass.UNDEFINED)
FIRST_PART = np.int32(SeparationClass.FIRST_PART)
SECOND_PART = np.int32(SeparationClass.SECOND_PART)
SEPARATOR = np.int32(SeparationClass.SEPARATOR)
