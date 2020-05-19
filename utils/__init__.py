import numpy as np

from .audio import *
from .files import *


def bytes_to_array(bytestring: bytes) -> np.array:
    return np.frombuffer(bytestring, dtype=np.int16)


def convert_to_bool(data):
    """Convert a 'true|false` string to a boolean."""
    upper_string = data.upper()
    if upper_string == "TRUE":
        return True
    elif upper_string == "FALSE":
        return False
    else:
        raise TypeError("Provided string is neither 'true' nor 'false'.")
