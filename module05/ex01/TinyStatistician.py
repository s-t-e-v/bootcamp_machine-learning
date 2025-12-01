import numpy as np


class TinyStatistician:
    @staticmethod
    def mean(x):
        """Compute the mean of a dataset."""
        row = TinyStatistician.validate_input(x)
        if row is None:
            return None
        return float(sum(row) / len(row))

    @staticmethod
    def median(x):
        """Compute the median of a dataset."""
        row = TinyStatistician.validate_input(x)
        if row is None:
            return None
        row = np.sort(row)
        length = len(row)
        mid_idx = length // 2

        if length % 2 !=0:
            return float(row[mid_idx])
        return float(row[mid_idx - 1] + row[mid_idx]) / 2

    @staticmethod
    def quartile(x):
        """Computes the 1st and 3rd quartiles of an 1D np array"""
        row = TinyStatistician.validate_input(x)
        if row is None:
            return None

        row = np.sort(row)
        length = len(row)

        if length < 2:
            return None
        
        split_idx = length // 2
        lower_half = row[:split_idx]
        upper_half = row[split_idx:] if length % 2 == 0\
                        else row[split_idx + 1:]
        
        q1 = TinyStatistician.median(lower_half)
        q3 = TinyStatistician.median(upper_half)
        return [float(q1), float(q3)]

    @staticmethod
    def validate_input(x):
        """Validate data and return it as a 1D numpy array copy."""
        v = x
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
            return None
        
        is_column_array = (len(v.shape) == 2 and v.shape[1] == 1)
        if not (len(v.shape) == 1 or is_column_array):
            return None

        if is_column_array:
            v = v.T[0]

        if not all(np.isreal(v) & ~np.isinf(v) & ~np.isnan(v)):
            return None

        return v.copy()
