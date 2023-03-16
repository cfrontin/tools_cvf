import numpy as np
import scipy.interpolate as interpol

from .windio_yaml_extractors import *


def get_collapsed_indep_axis(x_list: list) -> np.array:
    """
    combine samplings of independent variable

    take a list of independent variables and return a combination of all the
    unique values, in order; for inclusively collapsing samples of an
    independent variable s.t. they can be splined at all observed points

    inputs:
        - x_list: list of independent variable sampling arrays

    returns:
        - x_out: resulting array of independent variable
    """

    for idx, x in enumerate(x_list):
        x_list[idx] = x = np.array(x).flatten()
        assert len(x.shape) == 1, "indep. variable arrays must be 1D"

    x_out = np.stack(x_list)
    x_out.sort()
    x_out = np.unique(x_out)

    return x_out


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
