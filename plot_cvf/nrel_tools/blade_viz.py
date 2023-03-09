# system-level libraries
import sys
import os.path

# IO libraries
import argparse
import copy
import ruamel_yaml
# import yaml
from collections import OrderedDict

# plotting
import numpy as np
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import seaborn as sns
import plot_cvf
import pprint as pp


def check_input(input: list) -> list[str]:
    """
    check command line input files

    check command line input files to make sure they point to:
        - at least one real file
        - only yamls

    inputs:
        - input: list of CLI input files

    returns:
        - list of filenames to work on
    """

    # input checking: either string or list of strings
    if type(input) == str:
        input = [
            input,
        ]  # coerce to list when string is input
    assert type(input) == list, "entry must be a list of strings"
    assert len(input) > 0, "you need to specify at least one file"
    for x in input:
        assert type(x) == str, "entry must be a list of strings"

    # make sure inputs are legitimate yaml filenames
    filenames = []
    for filename in input:
        assert os.path.isfile(filename), "this code only works on yaml files"
        _, extension = os.path.splitext(filename)
        assert extension in [".yaml", ".yml"], "this code only works on yaml files"
        filenames.append(filename)

    return filenames

def load_yaml(filename):

    yaml= ruamel_yaml.YAML(typ= 'safe')
    with open(filename, 'r') as infile:
        data= yaml.load(infile)

    # # pyyaml version
    # # try to safe read
    # with open(filename, "r") as infile:
    # data = yaml.safe_load(infile)

    return data

def extract_blade_vectors(yaml_data: dict) -> tuple[list]:
    """
    given a yaml file, extract the blade design vectors:
        - airfoil_position_data: interp. map from nondim. span to airfoil types
        - chord_data: interp. map from nondim. span to chord
        - twist_data: interp. map from nondim. span to twist
        - pitch_axis_data: interp. map from nondim. span to pitch axis location
        - reference_axis_data: interp. map from nondim. span to reference axes

    inputs:
        - yaml_data: dictionary result from yaml import

    returns:
        - airfoil_position_data
        - chord_data
        - twist_data
        - pitch_axis_data
        - reference_axis_data
    """

    # get the data we will want to perturb
    airfoil_position_data = yaml_data["components"]["blade"]["outer_shape_bem"][
        "airfoil_position"
    ]
    chord_data = yaml_data["components"]["blade"]["outer_shape_bem"]["chord"]
    twist_data = yaml_data["components"]["blade"]["outer_shape_bem"]["twist"]
    pitch_axis_data = yaml_data["components"]["blade"]["outer_shape_bem"]["pitch_axis"]
    reference_axis_data = yaml_data["components"]["blade"]["outer_shape_bem"][
        "reference_axis"
    ]

    return (
        airfoil_position_data,
        chord_data,
        twist_data,
        pitch_axis_data,
        reference_axis_data,
    )


def extract_airfoil(yaml_data: dict, airfoil_label: str) -> dict:
    """
    extract the airfoil specification from a yaml file

    inputs:
        - yaml_data: dictionary result from yaml import
        - airfoil_label: airfoil code, must be in yaml_data['airfoils']

    returns:
        - airfoil_dict: deepcopy of the airfoil data in the yaml
    """

    airfoil_dict = {}

    # make sure the airfoil we want is there, get its index
    assert airfoil_label in [af["name"] for af in yaml_data["airfoils"]]
    idx_af = [af["name"] for af in yaml_data["airfoils"]].index(airfoil_label)

    # extract the yamldata, deepcopy
    airfoil_dict[airfoil_label] = copy.deepcopy(yaml_data["airfoils"][idx_af])

    return airfoil_dict


def get_splined_section(yaml_data: dict, zq: float) -> tuple:
    """
    get the splined version of the design variables at a given non-dim. span

    inputs:
        - yaml_data: dictionary result from yaml import
        - zq: nondimensional span at which to evaluate interpolation

    returns:
        - cq: chord at query point
        - twq: twist at query points
        - paq: pitch axis location at query point
        - _:
            - xq_refax: reference axis value at query point (in plane/cone of rotation, +: w/ airfoil motion)
            - yq_refax: reference axis value at query point (out of plane/cone of roration, +: rearward)
            - zq_refax: reference axis value at query point (span direction, +: outboard)
    """
    # get the data
    (
        _,
        chord_data,
        twist_data,
        pitch_axis_data,
        reference_axis_data,
    ) = extract_blade_vectors(yaml_data)

    # chord first, interpolate at a given query non-dim. span
    zp_chord = chord_data["grid"]
    cp_chord = chord_data["values"]
    cq = interpol.PchipInterpolator(zp_chord, cp_chord)(zq)

    # twist
    zp_twist = twist_data["grid"]
    twp_twist = twist_data["values"]
    twq = interpol.PchipInterpolator(zp_twist, twp_twist)(zq)

    # pitch axis
    zp_pitchax = pitch_axis_data["grid"]
    pap_pitchax = pitch_axis_data["values"]
    paq = interpol.PchipInterpolator(zp_pitchax, pap_pitchax)(zq)

    # reference axes
    ndsxp_refax = reference_axis_data["x"]["grid"]
    xp_refax = reference_axis_data["x"]["values"]
    xq_refax = interpol.PchipInterpolator(ndsxp_refax, xp_refax)(zq)
    ndsyp_refax = reference_axis_data["y"]["grid"]
    yp_refax = reference_axis_data["y"]["values"]
    yq_refax = interpol.PchipInterpolator(ndsyp_refax, yp_refax)(zq)
    ndszp_refax = reference_axis_data["z"]["grid"]
    zp_refax = reference_axis_data["z"]["values"]
    zq_refax = interpol.PchipInterpolator(ndszp_refax, zp_refax)(zq)

    return cq, twq, paq, (xq_refax, yq_refax, zq_refax)


def create_plot_splined(
    data_dict_list: list[dict], xlabel: str = None, ylabel: str = None
):
    """
    create a plot of a quantity interpreted via a PCHIP spline

    inputs:
        - data_dict_list: dict or list of dicts with keys:
            - x: x data to interpolate
            - y: y data to interpolate
            - label: label for data
        - xlabel: string for x label
        - ylabel: string for y label

    returns:
        - fig: pyplot figure for plot
        - ax: pyplot axis for plot
    """

    if type(data_dict_list) == dict:
        data_dict_list = [
            data_dict_list,
        ]
    Nplot = 201
    fig, ax = plt.subplots()
    for data_dict in data_dict_list:
        xp = data_dict["x"]
        yp = data_dict["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "-", label=data_dict["label"])
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    fig.legend()

    return fig, ax


def loft_foils(yaml_data: dict, zq: float, Ninterp: int = 101):
    """
    loft airfoils along the span between exactly specified airfoils

    possibly distinct airfoils are specified at a series of discrete non-dim.
    span locations, in order to loft the airfoil, use a two step process:
        1. put airfoils on a common grid (assumed points are arc-equispaced) by
            PCHIPS interpolation
        2. interpolate between distinct airfoils by interpolating between
            matched gridpoints across separate airfoils by PCHIPS interpolation

    inputs:
        - yaml_data: dictionary result from yaml import
        - zq: nondimensional span at which to evaluate interpolation
        - Ninterp: number of points at which to interpolate the airfoil data

    returns:
        - x_blended: lofted airfoil x data
        - y_blended: lofted airfoil y data
    """

    # get the spanwise data
    (
        airfoil_section_data,
        chord_data,
        twist_data,
        pitch_axis_data,
        reference_axis_data,
    ) = extract_blade_vectors(yaml_data)

    z_list = airfoil_section_data["grid"]
    foilname_list = airfoil_section_data["labels"]

    # airfoil
    af_interp_dict = {}
    for z_p, foilname_p in zip(z_list, foilname_list):
        af_here = extract_airfoil(yaml_data, foilname_p)[foilname_p]

        # get the airfoil coords
        x_coord = af_here["coordinates"]["x"]
        y_coord = af_here["coordinates"]["y"]

        # interpolate the airfoil spec onto a common number of points, assumed to be
        # equidistant in either angular or arc space
        basis_ref = np.linspace(0.0, 1.0, Ninterp)
        basis_coord = np.linspace(0.0, 1.0, len(x_coord))
        x_interp = np.zeros((Ninterp,))
        y_interp = np.zeros((Ninterp,))
        for i in range(Ninterp):
            x_interp[i] = interpol.PchipInterpolator(basis_coord, x_coord)(basis_ref[i])
            y_interp[i] = interpol.PchipInterpolator(basis_coord, y_coord)(basis_ref[i])

        af_interp_dict[z_p] = {
            "x": x_interp,
            "y": y_interp,
        }

    # now "interpolate" foil by interpolating x & y vars at a given index along span
    x_blended = np.zeros((Ninterp,))
    y_blended = np.zeros((Ninterp,))
    for i in range(Ninterp):
        # along a stringer
        x_stringer = np.array(
            [af_interp_dict[key]["x"][i] for key in sorted(af_interp_dict.keys())]
        )
        y_stringer = np.array(
            [af_interp_dict[key]["y"][i] for key in sorted(af_interp_dict.keys())]
        )
        z_stringer = np.array(sorted(af_interp_dict.keys()))
        x_blended[i] = interpol.PchipInterpolator(z_stringer, x_stringer)(zq)
        y_blended[i] = interpol.PchipInterpolator(z_stringer, y_stringer)(zq)

    return x_blended, y_blended


def do_blade_viz(
    data: dict,
    Nsection: int = 101,
    Ninterp: int = None,
    display_name=None,
    save_fig=True,
    show_fig=True,
):
    """
    vizualize in 3D a wind turbine blade or set of blades specified using a yaml
    file

    inputs:
        - data: a dictionary resulting from a yaml file input
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")  # get the airfoils

    # for airfoil_location, airfoil_label in zip(*airfoil_position_data.values()):
    #   print("%6f" % airfoil_location, airfoil_label)
    for airfoil_location in reversed(np.linspace(0.0, 1.0, Nsection)):
        # raw airfoil coordinate
        x_airfoil, y_airfoil = loft_foils(data, airfoil_location, Ninterp=Ninterp)

        # get splined data
        (
            chord_here,
            twist_here,
            pitch_axis_here,
            (x_refax, y_refax, z_refax),
        ) = get_splined_section(data, airfoil_location)

        # transform, translating pitch axis first
        x_airfoil -= pitch_axis_here
        # then multiplying by chord everywhere
        x_airfoil *= chord_here
        y_airfoil *= chord_here
        # rotate to get the airfoil to the right coords for the blade frame of ref.
        x_airfoil_old = copy.deepcopy(x_airfoil)
        y_airfoil_old = copy.deepcopy(y_airfoil)
        x_airfoil = y_airfoil_old
        y_airfoil = x_airfoil_old
        # now rotate by twist
        x_airfoil_old = copy.deepcopy(x_airfoil)
        y_airfoil_old = copy.deepcopy(y_airfoil)
        x_airfoil = +x_airfoil_old * np.cos(twist_here) + y_airfoil_old * np.sin(
            twist_here
        )
        y_airfoil = -x_airfoil_old * np.sin(twist_here) + y_airfoil_old * np.cos(
            twist_here
        )
        # now move to the reference axis
        x_airfoil += x_refax
        y_airfoil += y_refax

        plt.plot(x_airfoil, y_airfoil, z_refax * np.ones_like(x_airfoil))

    ax.axis("square")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.grid(False)

    # handle closeout appropriately
    if save_fig:
        filename_plot = "blade_%s.png" % display_name if display_name else "blade.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()


def do_design_comparison_plots(dataset: list[dict], show_fig=True, save_fig=None):
    """
    compare a wind turbine blade or set of blades in terms of the non-dim.
    spanwise variation of their design variables

    inputs:
        - dataset: a dict or list of dicts resulting from a yaml file input
    """

    # handle if a dictionary rather than list of dicts is passed
    if type(dataset) == dict:
        dataset = [
            dataset,
        ]

    chords_to_plot = []
    twists_to_plot = []
    refax_x_to_plot = []
    refax_y_to_plot = []
    refax_z_to_plot = []

    for display_name, data in dataset.items():
        # get the blade profile data
        (
            _,
            chord_data,
            twist_data,
            _,
            reference_axis_data,
        ) = extract_blade_vectors(data)

        # append things to plot
        chords_to_plot.append(
            {"x": chord_data["grid"], "y": chord_data["values"], "label": display_name}
        )
        twists_to_plot.append(
            {"x": twist_data["grid"], "y": twist_data["values"], "label": display_name}
        )
        refax_x_to_plot.append(
            {
                "x": reference_axis_data["x"]["grid"],
                "y": reference_axis_data["x"]["values"],
                "label": display_name,
            }
        )
        refax_y_to_plot.append(
            {
                "x": reference_axis_data["y"]["grid"],
                "y": reference_axis_data["y"]["values"],
                "label": display_name,
            }
        )
        refax_z_to_plot.append(
            {
                "x": reference_axis_data["z"]["grid"],
                "y": reference_axis_data["z"]["values"],
                "label": display_name,
            }
        )

    # make the plot
    fig, ax = create_plot_splined(
        chords_to_plot, xlabel="non-dimensional span", ylabel="chord"
    )
    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "designs_chord.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()

    fig, ax = create_plot_splined(
        twists_to_plot, xlabel="non-dimenstional span", ylabel="twist"
    )
    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "designs_twist.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()

    fig, ax = create_plot_splined(
        refax_x_to_plot, xlabel="non-dimenstional span", ylabel="ref. axis $x$"
    )
    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "designs_refax_x.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()

    fig, ax = create_plot_splined(
        refax_y_to_plot, xlabel="non-dimenstional span", ylabel="ref. axis $y$"
    )
    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "designs_refax_y.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()

    fig, ax = create_plot_splined(
        refax_z_to_plot, xlabel="non-dimenstional span", ylabel="ref. axis $z$"
    )
    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "designs_refax_z.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()


def main():
    ### parse CLI arguments
    parser = argparse.ArgumentParser(
        prog="blade_viz",
        description="cfrontin's automated wind turbine blade vizualization routines",
        epilog="have a nice day.\a\n",
    )
    parser.add_argument("-d", "--design", action="store_true", default=False)
    parser.add_argument("-b", "--blade", action="store_true", default=False)
    parser.add_argument("-n", "--noshow", action="store_true", default=False)
    parser.add_argument("-s", "--save", action="store_true", default=False)
    parser.add_argument("-l", "--latex", action="store_false", default=True)

    args, arg_filenames = parser.parse_known_args()

    ### do functionality

    # load the stylesheet for good plots
    plt.style.use(plot_cvf.get_stylesheets(dark=True, use_latex=args.latex))

    # get the filename after checking input for obvious issues
    filenames = check_input(arg_filenames)

    # create a set of plots to make, preserve ordering
    dataset = OrderedDict()

    # loop over filenames
    for filename in filenames:

        data= load_yaml(filename)

        # get the name based on the file
        display_name = os.path.split(os.path.splitext(filename)[0])[-1]

        # add this data to the dataset
        dataset[display_name] = data

        # while we're here do the airfoil viz
        if args.blade:
            do_blade_viz(
                data,
                101,
                51,
                display_name=display_name,
                show_fig=not args.noshow,
                save_fig=args.save,
            )

    # and also do comparison plots
    if args.design:
        do_design_comparison_plots(
            dataset, show_fig=not args.noshow, save_fig=args.save
        )


if __name__ == "__main__":
    main()
