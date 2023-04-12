import numpy as np
import pandas as pd

_BLADE_DAT_HEADER_LINES = 6
_BLADE_DAT_WIDTHS = [22, 23, 23, 23, 23, 23, 9]
_BLADE_DAT_COLUMN_NAMES = [
    "BlSpn",
    "BlCrvAC",
    "BlSwpAC",
    "BlCrvAng",
    "BlTwist",
    "BlChord",
    "BlAFID",
]


def load_blade_dat(filename: str) -> dict:
    """
    load a file given by `filename` and return its data

    inputs:
      - filename: a string filename that has already been checked

    outputs:
      - dict_out: a dictionary/list structure containing the data from the datafile
    """

    # read a datafile
    df = pd.read_fwf(
        filename,
        names=_BLADE_DAT_COLUMN_NAMES,
        header=_BLADE_DAT_HEADER_LINES,
        widths=_BLADE_DAT_WIDTHS,
    )
    df["ndspan"] = df.BlSpn / np.max(df.BlSpn)

    dict_out = {
        "nondim": df.ndspan.values,
        "span": df.BlSpn.values,  # m
        "blade_curve_ac": df.BlCrvAC.values,  # m, positive downstream hub radial axis
        "blade_sweep_ac": df.BlSwpAC.values,  # m, positive lagging hub radial axis
        "blade_curve_ang": df.BlCrvAng.values,  # deg, positive downstream of hub radial axis
        "blade_twist": df.BlTwist.values,  # deg, positive <- feather <- increase aoa
        "blade_chord": df.BlChord.values,  # m
    }

    return dict_out
