import copy


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
