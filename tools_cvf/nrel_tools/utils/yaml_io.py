import ruamel_yaml


def load_yaml(filename: str) -> dict:
    """
    load a file given by `filename` and return its data

    inputs:
        - filename: a string filename that has already been checked

    outputs:
        - data: a dictionary/list structure containing the data from the yaml
    """

    yaml = ruamel_yaml.YAML(typ="safe", pure=True)
    with open(filename, "r") as infile:
        data = yaml.load(infile)

    # # pyyaml version
    # # try to safe read
    # with open(filename, "r") as infile:
    # data = yaml.safe_load(infile)

    return data


def save_yaml(data: dict, filename_out: str) -> dict:
    """
    save a yaml data dictionary to a yaml file

    inputs:
        - data: a yaml data dictionary
        - filename_out: the yaml file destination to be written
    """

    yaml = ruamel_yaml.YAML(typ="safe", pure=True)
    with open(filename_out, "w") as outfile:
        yaml.dump(data, outfile)
