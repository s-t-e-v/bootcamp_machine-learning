import numpy as np


class TinyStatistician:
    def mean(x):
        x = TinyStatistician.validate_input(x)
        if not isinstance(x, np.ndarray):
            return None
        return sum(x) / x.shape[0]

    def median(x):
        x = TinyStatistician.validate_input(x)
        if not isinstance(x, np.ndarray):
            return None
        x.sort()
        m = x.shape[0]
        n = m // 2
        return float(x[n] if m % 2 != 0 else (x[n - 1] + x[n]) / 2)

    @staticmethod
    def validate_input(x):
        if isinstance(x, list):
            x = np.ndarray(x)
        if not isinstance(x, np.ndarray):
            return None
        if not (
            len(x.shape) == 1 or
                (len(x.shape) == 2 and x.shape[1] == 1)):
            return None
        if len(x.shape) == 2 and x.shape[1] == 1:
            x = x.T[0]
        if not all(np.isreal(x)):
            return None
        return x
