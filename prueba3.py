import numpy as np

def angle_error_from_arw(ARW, t_hours):
    """
    Error angular 1-sigma usando Angle Random Walk.

    Parameters
    ----------
    ARW : float
        Angle Random Walk [deg/sqrt(hour)].
    t_hours : float
        Tiempo en horas.

    Returns
    -------
    float
        Error angular 1-sigma [deg].
    """
    return ARW * np.sqrt(t_hours)


ARW = 0.14  # deg / sqrt(hour)

for t_hours in [0.5, 1, 2, 4]:
    print(t_hours, "h ->", angle_error_from_arw(ARW, t_hours), "deg")