"""Module containing the data classes for the geometry module."""

from typing import Dict

import numpy as np

PLOT_SETTINGS_SUBASSEMBLY = {
    "MP": {"color": "brown"},
    "TP": {"color": "goldenrod"},
    "TW": {"color": "grey"},
}


class Material(object):
    """Material derived from the raw data."""

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
    """Position of the components."""

    def __init__(
        self,
        x: np.float64 = np.float64(0.0),
        y: np.float64 = np.float64(0.0),
        z: np.float64 = np.float64(0.0),
        alpha: np.float64 = np.float64(0.0),
        beta: np.float64 = np.float64(0.0),
        gamma: np.float64 = np.float64(0.0),
        reference_system: str = "LAT",
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reference_system = reference_system
