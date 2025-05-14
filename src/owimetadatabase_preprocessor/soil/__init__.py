"""
OWI Meta Database Preprocessor - Soil Package

This package provides tools for processing, analyzing, and visualizing soil data
from the OWI Meta Database. It includes functionality for:

- Data retrieval and API interactions (io)
- Data processing and transformation (processing)
- Data visualization and plotting (visualization)

The package is organized into subpackages for each of these functional areas.
"""

# Import main classes for direct access
from .io import SoilAPI
from .processing import SoilDataProcessor, SoilprofileProcessor
from .visualization import SoilPlot

# Define the public API
__all__ = [
    # Core API classes
    "SoilAPI",
    "SoilDataProcessor",
    "SoilprofileProcessor",
    "SoilPlot",
    # Subpackages
    "io",
    "processing",
    "visualization",
]

# Import subpackages to make them available via soil.*
from . import io, processing, visualization
