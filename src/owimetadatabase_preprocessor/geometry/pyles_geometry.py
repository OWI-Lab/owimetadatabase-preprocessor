# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

import requests
import numpy as np
import pandas as pd
import json

from owimetadatabase_preprocessor.io import API


class GeometryPylesAPI(API):

    def get_subassemblies(self, projectsite=None, assetlocation=None, subassembly_type=None):
        """
        Get all structure subassemblies blocks for a given location
        :return: Dictionary with the following items:
           - 'data': Dataframe with the subassemblies
           - 'exists': Boolean determining whether any subassemblies were found
        """

        url_params = {}

        if projectsite is not None:
            url_params['asset__projectsite__title'] = projectsite

        if assetlocation is not None:
            url_params['asset__title'] = assetlocation

        if subassembly_type is not None:
            url_params['subassembly_type'] = subassembly_type

        sa = requests.get(
            '%s/geometry/userroutes/subassemblies' % self.api_root,
            headers=self.header,
            params=url_params)

        df = pd.DataFrame(json.loads(sa.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {
            'data': df,
            'exists': exists
        }

    def get_buildingblocks(self, projectsite=None, assetlocation=None, buildingblock_type=None):
        """
        Get all monopile building blocks for a given location
        :return: Dictionary with the following items:
           - 'data': Dataframe with the monopile building blocks
           - 'exists': Boolean determining whether any building blocks were found
        """

        url_params = {}

        if projectsite is not None:
            url_params['sub_assembly__asset__projectsite__title'] = projectsite

        if assetlocation is not None:
            url_params['sub_assembly__asset__title'] = assetlocation

        if buildingblock_type is not None:
            url_params['sub_assembly__subassembly_type'] = buildingblock_type

        bb = requests.get(
            '%s/geometry/userroutes/buildingblocks' % self.api_root,
            headers=self.header,
            params=url_params)

        df = pd.DataFrame(json.loads(bb.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {
            'data': df,
            'exists': exists
        }

    def get_monopile_pyles(self, water_depth, projectsite, assetlocation, cutoff_point=np.nan):
        """
        Returns a datafrome with the monopile geometry with the mudline as reference

        :param water_depth: Water depth in mLAT
        :param projectsite: Name of the project site
        :param assetlocation: Name of the wind turbine location
        :param cutoff_point: Elevation of the load application point in (mLAT) above the mudline
        :return:
        """
        # Retrieve the monopile cans
        bbs = self.get_buildingblocks(projectsite=projectsite, assetlocation=assetlocation,
                                      buildingblock_type='MP')
        # Retrieve the monopile subassembly
        sas = self.get_subassemblies(projectsite=projectsite, assetlocation=assetlocation,
                                     subassembly_type='MP')
        # Calculate the pile penetration
        toe_depth_lat = sas['data']['z_position'].iloc[0]
        penetration = -((1e-3 * toe_depth_lat) - water_depth)

        # Create the pile for subsequent response analysis
        pile = pd.DataFrame()

        for i, row in bbs['data'].iterrows():
            if i != 0:
                pile.loc[i, "Depth to [m]"] = penetration - 1e-3 * bbs['data'].loc[i-1, 'z_position']
                pile.loc[i, "Depth from [m]"] = penetration - 1e-3 * row['z_position']
                pile.loc[i, "Pile material"] = row["material_name"]
                pile.loc[i, "Pile material submerged unit weight [kN/m3]"] = 1e-2 * row["density"] - 10
                pile.loc[i, "Wall thickness [mm]"] = row['wall_thickness']
                pile.loc[i, "Diameter [m]"] = 1e-3 * 0.5 * (row['bottom_outer_diameter'] + row['top_outer_diameter'])
                pile.loc[i, "Youngs modulus [GPa]"] = row['youngs_modulus']
                pile.loc[i, "Poissons ratio [-]"] = row['poissons_ratio']

        pile.sort_values('Depth from [m]', inplace=True)
        pile.reset_index(drop=True, inplace=True)

        # Cut off at the mudline
        if not np.math.isnan(cutoff_point):
            pile = pile.loc[pile["Depth to [m]"] > cutoff_point].reset_index(drop=True)
            pile.loc[0, 'Depth from [m]'] = cutoff_point

        return pile