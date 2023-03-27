import os.path
import argparse

import numpy as np
import scipy.interpolate as interpol

# import yaml
import ruamel_yaml as ryaml


def main():
    ### parse CLI arguments
    parser = argparse.ArgumentParser(
        prog="yaml_stretcher",
        description="cfrontin's windIO yaml re-interpolator code",
        epilog="have a nice day.\a\n",
    )
    parser.add_argument("src", help="file to re-interpolate")
    parser.add_argument("-d", "--dest", default=None, help="filename to output")
    parser.add_argument(
        "-n",
        "--Ninterp",
        type=int,
        default=150,
        help="target number of interpolation points",
    )

    args = parser.parse_args()

    ### inputs

    # specify the file to ping and the file that should result
    filename_in = args.src
    filename_out = (
        args.dest
        if args.dest is not None
        else ("%s_pchips%d.yaml" % (os.path.splitext(args.src)[0], args.Ninterp))
    )

    # number of points to interpolate onto
    Ninterp = args.Ninterp

    ### kernel

    # open the BEM optimum file
    yaml = ryaml.YAML(typ="safe", pure=True)
    with open(filename_in, "r") as infile_yaml:
        # # pyyaml loader
        # data= yaml.safe_load(infile_yaml)

        data = yaml.load(infile_yaml)

    # the data we're interested in is the aero. spec. of the airfoils
    data_of_interest = [
        data["components"]["blade"]["outer_shape_bem"]["chord"],
        data["components"]["blade"]["outer_shape_bem"]["twist"],
        data["components"]["blade"]["outer_shape_bem"]["pitch_axis"],
        data["components"]["blade"]["outer_shape_bem"]["reference_axis"]["x"],
        data["components"]["blade"]["outer_shape_bem"]["reference_axis"]["y"],
        data["components"]["blade"]["outer_shape_bem"]["reference_axis"]["z"],
    ]

    # new span points: equal distribution on the domain
    x_new = np.linspace(0.0, 1.0, Ninterp)

    for datum in data_of_interest:
        x_orig = datum["grid"]
        y_orig = datum["values"]

        interpolator = interpol.PchipInterpolator(x_orig, y_orig)
        y_new = interpolator(x_new)

        datum["grid"] = x_new.tolist()
        datum["values"] = y_new.tolist()

    # write the perturbed data to a new file
    with open(filename_out, "w") as outfile_yaml:
        yaml.dump(data, outfile_yaml)


if __name__ == "__main__":
    main()
