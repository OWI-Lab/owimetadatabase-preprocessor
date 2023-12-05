"""Module containing the data classes for the geometry module."""

from typing import Dict, Union

import json
import pandas as pd
import requests


PLOT_SETTINGS_SUBASSEMBLY = {
    "MP": {"color": "brown"},
    "TP": {"color": "goldenrod"},
    "TW": {"color": "grey"},
}


class Material(object):
    """Materials derived from the raw data."""

    def __init__(self, json: Dict[str, str]) -> None:
        self.title = json["title"]
        self.description = json["description"]
        self.density = json["density"]
        self.poisson_ratio = json["poisson_ratio"]
        self.young_modulus = json["young_modulus"]
        self.id = json["id"]

    def as_dict(self) -> Dict[str, str]:
        """Transform data into dictionary.
        
        :return: Dictionary with the following keys:

            - "title": Name of the material.
            - "description": Description of the material.
            - "poisson_ratio": Poisson ratio of the material.
            - "young_modulus": Young modulus of the material.
        """
        return {
            "title": self.title,
            "description": self.description,
            "poisson_ratio": self.poisson_ratio,
            "young_modulus": self.young_modulus,
        }


class Position(object):
    def __init__(self, x=0, y=0, z=0, alpha=0, beta=0, gamma=0, reference_system="LAT"):
        self.x = x
        self.y = y
        self.z = z

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.reference_system = reference_system
