import numpy as np

from .audio import *
from .files import *


def bytes_to_array(bytestring: bytes) -> np.array:
    return np.frombuffer(bytestring, dtype=np.int16)
