#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Utiliary methods to e.g. access simulation parameters."""

_PARAMETERS = {
        # simulation
        "obstacle_radius_factor": 0.0625,
        "smoke_source_radius_factor": 0.0725,
        "resolution": 32,
        "relative_x_position": 0.25,
        "velocity_in": (0.9, 0, 0),
        "nr_frames": 300,
        "y_position_min": 2,
        "y_position_max": 30,
        # post-processing
        "downscaling_factors": (0.25, 0.25, 1.0),
        # manta-related
        "show_gui": False,
        }

def get_parameter(name):
    """Retrieve simulation parameter. Raises exception is specified name not
    found."""
    try:
        return _PARAMETERS[name]
    except KeyError as e:
        print("Unknown parameter '{}'. Choose one of {}.".format(
            name, ", ".join(_PARAMETERS.keys())))
        raise e
