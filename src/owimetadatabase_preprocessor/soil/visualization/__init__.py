"""
Visualization module for soil data in the OWI Meta Database.

This module provides tools for visualizing soil data, including:
- Soil profile fence diagrams
- CPT fence diagrams
- Combined fence diagrams with both soil profiles and CPT data
- Maps of soil test locations

The main interface is through the SoilPlot class, which creates
Plotly figures from soil data retrieved via the SoilAPI.
"""

from .soil_visualizer import SoilPlot

__all__ = ["SoilPlot"]
