# system-level libraries
import sys
import os
import os.path
import glob

# IO libraries
import argparse
import copy

# import yaml
from collections import OrderedDict

# plotting
import numpy as np
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tools_cvf
import pprint as pp

from tools_cvf.nrel_tools.utils import *

blade_viz_image_formats = ("png",)
blade_viz_image_prefixes = (
    "blade_",
    "designs_",
    "planform_",
)


def create_plot_splined(
    data_dict_list: list[dict],
    xlabel: str = None,
    ylabel: str = None,
    figax: tuple[plt.figure, plt.axis] = None,
    figsize: tuple[float, float] = None,
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
    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        assert (
            len(figax) == 2
        ), "to use existing figure, submit (fig, ax) tuple as figax"
        assert (
            type(figax) == tuple
        ), "to use existing figure, submit (fig, ax) tuple as figax"
        assert (
            type(figax[0]) == mpl.figure.Figure
        ), "to use existing figure, submit (fig, ax) tuple as figax"
        assert (
            type(figax[1]) == plt.Axes
        ), "to use existing figure, submit (fig, ax) tuple as figax"
        # plt.figure(figax[0])
        # plt.sca(figax[1])
        fig, ax = figax
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


def do_design_comparison_plots_windio(
    dataset: list[dict],
    show_fig=True,
    save_fig=None,
    figsize=None,
):
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
        chords_to_plot,
        xlabel="non-dimensional span",
        ylabel="chord",
        figsize=figsize,
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
        twists_to_plot,
        xlabel="non-dimenstional span",
        ylabel="twist",
        figsize=figsize,
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
        refax_x_to_plot,
        xlabel="non-dimenstional span",
        ylabel="ref. axis $x$",
        figsize=figsize,
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
        refax_y_to_plot,
        xlabel="non-dimenstional span",
        ylabel="ref. axis $y$",
        figsize=figsize,
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
        refax_z_to_plot,
        xlabel="non-dimenstional span",
        ylabel="ref. axis $z$",
        figsize=figsize,
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


def do_planform_comparison_plots_windio(
    dataset: list[dict], twist_correct=True, show_fig=True, save_fig=None, figsize=None,
):
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

    length_le_to_plot = []
    length_te_to_plot = []
    centerline_to_plot = []

    for display_name, data in dataset.items():
        # get the blade profile data
        (
            _,
            chord_data,
            twist_data,
            pitchax_data,
            reference_axis_data,
        ) = extract_blade_vectors(data)

        # for now, just make sure that the grids are the same
        assert np.all(chord_data["grid"] == twist_data["grid"])
        assert np.all(chord_data["grid"] == pitchax_data["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["x"]["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["y"]["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["z"]["grid"])
        # TODO: make an independent variable spline mutualizer

        # correct for twist, it's effect on top-down planform
        twist_correct_term = (
            np.cos(twist_data["values"])
            if twist_correct
            else np.ones(twist_data["grid"])
        )

        length_le_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": twist_correct_term
                * np.asarray(chord_data["values"])
                * np.asarray(pitchax_data["values"]),
                "label": display_name,
            }
        )
        length_te_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": -twist_correct_term
                * np.asarray(chord_data["values"])
                * (1.0 - np.asarray(pitchax_data["values"])),
                "label": display_name,
            }
        )
        centerline_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": np.asarray(reference_axis_data["y"]["values"]),
                "label": display_name,
            }
        )

    # make the plot
    Nplot = 201
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([], [], ".-", color="gray", label="centerline")
    ax.plot([], [], "--", color="gray", label="leading edge")
    ax.plot([], [], "-", color="gray", label="trailing edge")

    for data_edges in zip(*(centerline_to_plot, length_le_to_plot, length_te_to_plot)):
        # centerline
        xp = data_edges[0]["x"]
        yp = data_edges[0]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "-.", label=data_edges[0]["label"])
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

        # leading edge
        xp = data_edges[1]["x"]
        yp = data_edges[1]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "--", c=pt0[-1].get_color(), label="_")
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

        # trailing edge
        xp = data_edges[2]["x"]
        yp = data_edges[2]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "-", c=pt0[-1].get_color(), label="_")
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

    ax.set_xlabel("non-dim. span")
    ax.set_ylabel(
        "planform %s" % ("(twist-corrected)" if twist_correct else "(pre-twist)")
    )
    ax.grid(False)
    fig.legend()

    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "planform_lengths.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()
    plt.show()


def do_raw_plots_aerodyn(dataset: list[dict], show_fig=True, save_fig=None):
    """
    plot what aerodyn sees
    """

    def plot_func(dataset, yvar, ylab, outfile_var):
        fig, ax = plt.subplots()
        for display_name, data in dataset.items():
            pt0 = ax.plot(data["nondim"], data[yvar], ".-", label=display_name)
        ax.set_xlabel("non-dim. span")
        ax.set_ylabel(ylab)
        ax.grid(False)
        fig.legend()

        # handle figure closeout appropriately
        if save_fig:
            filename_plot = "view_aerodyn_%s.png" % outfile_var
            plt.savefig(filename_plot, bbox_inches="tight")

    plot_func(dataset, "blade_chord", "\\texttt{BlChord}", "chord")
    plot_func(dataset, "blade_twist", "\\texttt{BlTwist}", "twist")
    plot_func(dataset, "blade_sweep_ac", "\\texttt{BlSwpAC}", "sweep_ac")
    plot_func(dataset, "blade_curve_ac", "\\texttt{BlCrvAC}", "curve_ac")
    plot_func(dataset, "span", "\\texttt{BlSpn}", "span")

    # show or close
    if show_fig:
        plt.show()
    else:
        plt.close("all")


def do_design_comparison_plots_aerodyn(
    dataset: list[dict], show_fig=True, save_fig=None
):
    """
    compare a wind turbine blade or set of blades in terms of the non-dim.
    spanwise variation of their design variables

    inputs:
        - dataset: a dict or list of dicts resulting from a blade datafile input
    """

    # handle if a dictionary rather than list of dicts is passed
    if type(dataset) == dict:
        dataset = [
            dataset,
        ]

    chords_to_plot = []
    twists_to_plot = []
    refax_x_to_plot = []  # in this case, aligned with aerodynamic center
    refax_y_to_plot = []  # in this case, aligned with aerodynamic center
    refax_z_to_plot = []  # in this case, aligned with aerodynamic center

    for display_name, data in dataset.items():
        # get the blade profile data
        chord_data = {"grid": data["nondim"], "values": data["blade_chord"]}
        twist_data = {"grid": data["nondim"], "values": data["blade_twist"]}
        reference_axis_data = {
            "x": {
                "grid": data["nondim"],
                "values": [-val for val in data["blade_sweep_ac"]],
            },  # in plane of rotation, positive leading
            "y": {
                "grid": data["nondim"],
                "values": data["blade_curve_ac"],
            },  # coaxial with hub, positive downstream
            "z": {"grid": data["nondim"], "values": data["span"]},
        }

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


def do_planform_comparison_plots_aerodyn(
    dataset: list[dict], twist_correct=True, show_fig=True, save_fig=None
):
    """
    compare a wind turbine blade or set of blades in terms of the non-dim.
    spanwise variation of their design variables

    inputs:
        - dataset: a dict or list of dicts resulting from a blade datafile input
    """

    # handle if a dictionary rather than list of dicts is passed
    if type(dataset) == dict:
        dataset = [
            dataset,
        ]

    length_le_to_plot = []
    length_te_to_plot = []
    centerline_to_plot = []

    for display_name, data in dataset.items():
        # get the blade profile data
        chord_data = {"grid": data["nondim"], "values": data["blade_chord"]}
        twist_data = {"grid": data["nondim"], "values": data["blade_twist"]}
        reference_axis_data = {
            "x": {
                "grid": data["nondim"],
                "values": [-val for val in data["blade_sweep_ac"]],
            },  # in plane of rotation, positive leading
            "y": {
                "grid": data["nondim"],
                "values": data["blade_curve_ac"],
            },  # coaxial with hub, positive downstream
            "z": {"grid": data["nondim"], "values": data["span"]},
        }

        raise NotImplementedError("pick up the implementation from here. -cfrontin")

        # for now, just make sure that the grids are the same
        assert np.all(chord_data["grid"] == twist_data["grid"])
        assert np.all(chord_data["grid"] == pitchax_data["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["x"]["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["y"]["grid"])
        assert np.all(chord_data["grid"] == reference_axis_data["z"]["grid"])
        # TODO: make an independent variable spline mutualizer

        # correct for twist, it's effect on top-down planform
        twist_correct_term = (
            np.cos(twist_data["values"])
            if twist_correct
            else np.ones(twist_data["grid"])
        )

        length_le_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": twist_correct_term
                * np.asarray(chord_data["values"])
                * np.asarray(pitchax_data["values"]),
                "label": display_name,
            }
        )
        length_te_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": -twist_correct_term
                * np.asarray(chord_data["values"])
                * (1.0 - np.asarray(pitchax_data["values"])),
                "label": display_name,
            }
        )
        centerline_to_plot.append(
            {
                "x": chord_data["grid"],
                "y": np.asarray(reference_axis_data["y"]["values"]),
                "label": display_name,
            }
        )

    # make the plot
    Nplot = 201
    fig, ax = plt.subplots()
    ax.plot([], [], "w.-", label="centerline")
    ax.plot([], [], "w--", label="leading edge")
    ax.plot([], [], "w-", label="trailing edge")

    for data_edges in zip(*(centerline_to_plot, length_le_to_plot, length_te_to_plot)):
        # centerline
        xp = data_edges[0]["x"]
        yp = data_edges[0]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "-.", label=data_edges[0]["label"])
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

        # leading edge
        xp = data_edges[1]["x"]
        yp = data_edges[1]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "--", c=pt0[-1].get_color(), label="_")
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

        # trailing edge
        xp = data_edges[2]["x"]
        yp = data_edges[2]["y"]
        x = np.linspace(np.min(xp), np.max(xp), Nplot)
        y = interpol.PchipInterpolator(xp, yp)(x)
        pt0 = ax.plot(x, y, "-", c=pt0[-1].get_color(), label="_")
        ax.plot(xp, yp, ".", c=pt0[-1].get_color(), label="_")

    ax.set_xlabel("non-dim. span")
    ax.set_ylabel(
        "planform %s" % ("(twist-corrected)" if twist_correct else "(pre-twist)")
    )
    ax.grid(False)
    fig.legend()

    # handle figure closeout appropriately
    if save_fig:
        filename_plot = "planform_lengths.png"
        plt.savefig(filename_plot, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()


def do_windio_bladeviz(args, arg_filenames, figsize=None):
    # get the filename after checking input for obvious issues
    filenames = check_yaml_files(arg_filenames)

    # create a set of plots to make, preserve ordering
    dataset = OrderedDict()

    # loop over filenames
    for filename in filenames:
        data = load_yaml(filename)

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
        do_design_comparison_plots_windio(
            dataset, show_fig=not args.noshow, save_fig=args.save, figsize=figsize
        )

    # and also do planform plots
    if args.planform:
        do_planform_comparison_plots_windio(
            dataset, show_fig=not args.noshow, save_fig=args.save, figsize=figsize
        )


def do_aerodyn_bladeviz(args, arg_filenames):
    # get the filename after checking input for obvious issues
    filenames = check_blade_dat_files(arg_filenames)

    # create a set of plots to make, preserve ordering
    dataset = OrderedDict()

    for filename in filenames:
        data = load_blade_dat(filename)

        # get the name based on the file
        display_name = os.path.split(os.path.splitext(filename)[0])[-1]

        # add this data to the dataset
        dataset[display_name] = data

    do_raw_plots_aerodyn(dataset, show_fig=not args.noshow, save_fig=args.save)

    # do comparison plots
    if args.design:
        do_design_comparison_plots_aerodyn(
            dataset, show_fig=not args.noshow, save_fig=args.save
        )


def main():
    ### parse CLI arguments
    parser = argparse.ArgumentParser(
        prog="blade_viz",
        description="cfrontin's automated wind turbine blade vizualization routines",
        epilog="have a nice day.\a\n",
    )
    parser.add_argument("-d", "--design", action="store_true", default=False)
    parser.add_argument("-p", "--planform", action="store_true", default=False)
    parser.add_argument("-b", "--blade", action="store_true", default=False)
    parser.add_argument("-n", "--noshow", action="store_true", default=False)
    parser.add_argument("-s", "--save", action="store_true", default=False)
    parser.add_argument("-l", "--latex", action="store_false", default=True)
    parser.add_argument("--small", action="store_true", default=False)
    parser.add_argument("--paper", action="store_true", default=False)
    parser.add_argument("--aerodyn", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)

    args, arg_filenames = parser.parse_known_args()

    ### do functionality

    if args.clean:
        for file in glob.glob("*"):
            if not file.endswith(blade_viz_image_formats):
                continue
            if not file.startswith(blade_viz_image_prefixes):
                continue
            os.remove(file)

        return

    # load the stylesheet for good plots
    plt.style.use(tools_cvf.get_stylesheets(
        dark=not args.paper,
        use_latex=args.latex
    ))

    if args.aerodyn:
        do_aerodyn_bladeviz(args, arg_filenames)
    else:
        do_windio_bladeviz(args, arg_filenames, figsize=(5,4) if args.small else None)


if __name__ == "__main__":
    main()
