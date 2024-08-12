import warnings
from scipy.interpolate import interp1d


def relative_resolution(mtf, x: float = 50) -> float:
    """Return the line pair value at the given rMTF resolution value.

    Parameters
    ----------
    mtf: array-like
        The MTF values to calculate the relative resolution of.
    x : float
        The percentage of the rMTF to determine the line pair value. Must be between 0 and 100.
    """
    f = interp1d(
        list(mtf.norm_mtfs.values()),
        list(mtf.norm_mtfs.keys()),
        fill_value="extrapolate",
    )
    mtf_res = f(x / 100)
    if mtf_res > max(mtf.spacings):
        warnings.warn(
            f"MTF resolution wasn't calculated for {x}% that was asked for. The value returned is an extrapolation. Use a higher % MTF to get a non-interpolated value."
        )
    return float(mtf_res)


def ratio(a, b):
    """
    Calculate the relative difference between two numbers.
    """
    if a == b:
        return 0
    try:
        return (a - b) / b
    except ZeroDivisionError:
        return float('inf')