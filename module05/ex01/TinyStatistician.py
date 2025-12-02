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

        if length % 2 != 0:
            return float(row[mid_idx])
        return float(row[mid_idx - 1] + row[mid_idx]) / 2

    @staticmethod
    def quartile(x):
        """Computes the 1st and 3rd quartiles of an 1D np array"""
        row = TinyStatistician.validate_input(x)
        if row is None:
            return None

        row = np.sort(row)

        quartiles = [
            TinyStatistician._percentile_from_sorted_array(row, p)
            for p in [25, 75]
        ]
        return quartiles

    @staticmethod
    def percentile(x, p):
        """Computes the expected percentile"""
        row = TinyStatistician.validate_input(x)
        if row is None:
            return None
        if not isinstance(p, (int, float)) or not 0 <= p <= 100:
            return None
        row = np.sort(row)
        return TinyStatistician._percentile_from_sorted_array(row, p)

    @staticmethod
    def _percentile_from_sorted_array(x, percentile):
        """Compute percentile using linear interpolation.

        Internal helper - assumes x is already validated and sorted.
        Use percentile() for the public API.
        """
        length = len(x)

        position = float(percentile / 100 * (length - 1))
        if position.is_integer():
            return float(x[int(position)])
        low_idx = int(position)
        high_idx = low_idx + 1
        weight = position % 1
        return float(
            x[low_idx] + (x[high_idx] - x[low_idx]) * weight
        )

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

        if v.size == 0:
            return None

        if not np.all(np.isfinite(v)):
            return None

        return v.copy()
