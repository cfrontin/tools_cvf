import os.path


def check_yaml_files(input: list) -> list[str]:
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


def check_blade_dat_files(input: list) -> list[str]:
    """
    check command line input files of the *_blade.dat variety

    check command line input files to make sure they point to:
        - at least one real file
        - only *_blade.dat files

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

    # make sure inputs are legitimate *_blade.dat filenames
    filenames = []
    for filename in input:
        assert os.path.isfile(filename), "this code only works on files that exist"
        basename, extension = os.path.splitext(filename)
        assert extension in [
            ".dat",
        ], "this code only works on .dat files"
        assert basename.endswith("_blade"), "the input files should end in '_blade'"
        filenames.append(filename)

    return filenames
