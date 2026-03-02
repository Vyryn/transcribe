from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    """Compute a percentile using linear interpolation.

    Parameters
    ----------
    values : list[float]
        Input values.
    p : float
        Percentile in the range ``[0.0, 1.0]``.

    Returns
    -------
    float
        Interpolated percentile value, or ``0.0`` when input is empty.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
