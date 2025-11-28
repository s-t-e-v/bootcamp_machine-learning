import numpy as np


class TinyStatistician:
    def mean(x):
        if isinstance(x, list):
            x = np.ndarray(x)
        if not isinstance(x, np.ndarray):
            return None
        return sum(x) / x.shape[0]
