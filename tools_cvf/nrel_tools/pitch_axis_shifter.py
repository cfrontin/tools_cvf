# system-level libraries
import sys
import os.path

# IO libraries
import argparse
import copy

# import yaml
from collections import OrderedDict

# plotting
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpol
import seaborn as sns
import pprint as pp

from tools_cvf.nrel_tools.utils import *


def transform(
    refax_x_in: np.array,
    refax_y_in: np.array,
    refax_z_in: np.array,
    pitchax_in: np.array,
    chord_in: np.array,
    twist_in: np.array,
    pitchax_out: np.array = None,
):
    """
    transform a blade geometry, given by numpy arrays, to have a new pitch axis
    by moving the reference axis appropriately

    inputs:
        - refax_x_in: \
        - refax_y_in: | -> the reference axis that we start with
        - refax_z_in: /
        - pitchax_in: the pitch axis locations we start with
        - chord_in: the starting chords
        - twist_in: the starting twists
        - pitchax_out: the pitch axis location we want to end up with
            (defaults to zeros, meaning the leading edge)
    """

    # start with the output the same as the output, except the pitch axis
    if pitchax_out is None:
        pitchax_out = np.zeros_like(pitchax_in)
    refax_z_out = np.array(refax_z_in)
    refax_x_out = np.array(refax_x_in)
    refax_y_out = np.array(refax_y_in)
    chord_out = np.array(chord_in)
    twist_out = np.array(twist_in)

    refax_x_out += np.array(pitchax_in) * chord_out * np.sin(twist_out)
    refax_y_out += np.array(pitchax_in) * chord_out * np.cos(twist_out)

    # make sure the root is centered for mounting on the hub
    pitchax_out[0:2] = pitchax_in[0:2]
    refax_x_out[0:2] = refax_x_in[0:2]
    refax_y_out[0:2] = refax_y_in[0:2]
    refax_z_out[0:2] = refax_z_in[0:2]
    chord_out[0:2] = chord_in[0:2]
    twist_out[0:2] = twist_in[0:2]

    return refax_x_out, refax_y_out, refax_z_out, pitchax_out, chord_out, twist_out


def shifter(
    filename_yaml_in: str, filename_yaml_out: str = None, pitch_ax_des: np.array = None
):
    """
    take in a valid yaml file location, then output a new yaml file that gives
    a new blade with the reference axis shifted s.t. twist is applied about
    a new pitch axis but the resulting blade is geometrically equivalent

    for now it also does plots, for debugging

    inputs:
        - filename_yaml_in: filename for input yaml
        - filename_yaml_out: filename for output yaml
        - pitch_ax_des: new pitch axis to be used (default zero: leading edge)
    """

    # check the files and package the right way
    filenames_yaml = check_yaml_files(filename_yaml_in)
    # make sure its a list of one and repack
    assert (type(filenames_yaml) == list) and (len(filenames_yaml) == 1)
    filename_yaml_in = filenames_yaml[0]

    # extract the data
    data = load_yaml(filename_yaml_in)
    data_new = copy.deepcopy(data)

    _, chord_data, twist_data, pitchax_data, refax_data = extract_blade_vectors(data)
    _, chord_data2, twist_data2, pitchax_data2, refax_data2 = extract_blade_vectors(
        data_new
    )

    (
        refax_data2["x"]["values"],
        refax_data2["y"]["values"],
        refax_data2["z"]["values"],
        pitchax_data2["values"],
        chord_data2["values"],
        twist_data2["values"],
    ) = transform(
        refax_data["x"]["values"],
        refax_data["y"]["values"],
        refax_data["z"]["values"],
        pitchax_data["values"],
        chord_data["values"],
        twist_data["values"],
    )

    refax_data2["x"]["values"] = refax_data2["x"]["values"].tolist()
    refax_data2["y"]["values"] = refax_data2["y"]["values"].tolist()
    refax_data2["z"]["values"] = refax_data2["z"]["values"].tolist()
    pitchax_data2["values"] = pitchax_data2["values"].tolist()
    chord_data2["values"] = chord_data2["values"].tolist()
    twist_data2["values"] = twist_data2["values"].tolist()

    refax_y_interpolator = interpol.PchipInterpolator(
        refax_data["y"]["grid"], refax_data["y"]["values"]
    )
    refax_z_interpolator = interpol.PchipInterpolator(
        refax_data["z"]["grid"], refax_data["z"]["values"]
    )
    pitchax_interpolator = interpol.PchipInterpolator(
        pitchax_data["grid"], pitchax_data["values"]
    )

    refax_y_interpolator2 = interpol.PchipInterpolator(
        refax_data2["y"]["grid"], refax_data2["y"]["values"]
    )
    refax_z_interpolator2 = interpol.PchipInterpolator(
        refax_data2["z"]["grid"], refax_data2["z"]["values"]
    )
    pitchax_interpolator2 = interpol.PchipInterpolator(
        pitchax_data2["grid"], pitchax_data2["values"]
    )

    # for now assume that all the grids are the same
    assert np.all(refax_data["x"]["grid"] == refax_data["y"]["grid"])
    assert np.all(refax_data["x"]["grid"] == refax_data["z"]["grid"])
    assert np.all(refax_data["x"]["grid"] == chord_data["grid"])
    assert np.all(refax_data["x"]["grid"] == twist_data["grid"])

    # # make sure new data is stored as a list now
    # for datum in [chord_data2, twist_data2, pitchax_data2]:
    #     for key in ['grid', 'values']:
    #         if type(datum[key]) == np.array:
    #             datum[key]= datum[key].tolist()
    # for coord in refax_data2.keys():
    #     for key in ['grid', 'values']:
    #         if type(refax_data2[coord][key]) == np.array:
    #             refax_data2[coord][key]= refax_data2[coord][key].tolist()

    # output the new yaml
    if not filename_yaml_out is None:
        save_yaml(data_new, filename_yaml_out)

    plt.subplots()
    plt.plot([], [], "b-", label="baseline")
    plt.plot(refax_data["z"]["values"], refax_data["y"]["values"], "b--", label="_")
    plt.plot(
        refax_z_interpolator(chord_data["grid"]),
        refax_y_interpolator(chord_data["grid"])
        + pitchax_interpolator(chord_data["grid"]) * chord_data["values"],
        "b-",
        label="_",
    )
    plt.plot(
        refax_z_interpolator(chord_data["grid"]),
        refax_y_interpolator(chord_data["grid"])
        - (1 - pitchax_interpolator(chord_data["grid"])) * chord_data["values"],
        "b-",
        label="_",
    )
    plt.plot(
        refax_z_interpolator2(chord_data2["grid"]),
        refax_y_interpolator2(chord_data2["grid"])
        + pitchax_interpolator2(chord_data2["grid"]) * chord_data2["values"],
        "r-.",
        label="_",
    )
    plt.plot(
        refax_z_interpolator2(chord_data2["grid"]),
        refax_y_interpolator2(chord_data2["grid"])
        - (1 - pitchax_interpolator2(chord_data2["grid"])) * chord_data2["values"],
        "r-.",
        label="_",
    )
    plt.plot(refax_data2["z"]["values"], refax_data2["y"]["values"], "g:", label="_")
    plt.axis("equal")
    plt.show()


def main():
    pitch_ax_des = 0.0
    filename_yaml_in = os.path.join(
        os.path.split(__file__)[0], "../../examples/BEM_Optimum.yaml"
    )
    filename_yaml_out = os.path.join(
        os.path.split(__file__)[0], "../../examples/BEM_opt_modref0.yaml"
    )

    shifter(
        filename_yaml_in, filename_yaml_out=filename_yaml_out, pitch_ax_des=pitch_ax_des
    )


if __name__ == "__main__":
    main()
