"""
Processing module for soil data in the OWI Meta Database.

This module provides tools for processing soil data and soil profiles,
including coordinate transformation, data processing for in-situ tests,
and preparation of soil profiles for various soil-structure interaction models.
"""

from .soil_pp import SoilDataProcessor, SoilprofileProcessor

__all__ = ["SoilDataProcessor", "SoilprofileProcessor"]
