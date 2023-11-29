"""Module to connect to the database API to retrieve and operate on geometry data."""

from owimetadatabase_preprocessor.io import API


class GeometryAPI(API):
    def get_subassemblies(
        self, projectsite=None, assetlocation=None, subassembly_type=None
    ):
        """
        Get all structure subassemblies blocks for a given location
        :return:
        """
        url_params = {}
        if projectsite is not None:
            url_params["asset__projectsite__title"] = projectsite
        if assetlocation is not None:
            url_params["asset__title"] = assetlocation
        if subassembly_type is not None:
            url_params["subassembly_type"] = subassembly_type
        url_data_type = "/geometry/userroutes/subassemblies"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_buildingblocks(
        self,
        projectsite=None,
        assetlocation=None,
        buildingblock_type=None,
        subassembly_id=None,
    ):
        """
        Get all building blocks for a given location
        :return:
        """
        url_params = {}
        if projectsite is not None:
            url_params["sub_assembly__asset__projectsite__title"] = projectsite
        if assetlocation is not None:
            url_params["sub_assembly__asset__title"] = assetlocation
        if buildingblock_type is not None:
            url_params["sub_assembly__subassembly_type"] = buildingblock_type
        if subassembly_id is not None:
            url_params["sub_assembly__id"] = subassembly_id
        url_data_type = "/geometry/userroutes/buildingblocks"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_materials(
        self,
        projectsite=None,
        assetlocation=None,
        buildingblock_type=None,
        subassembly_id=None,
    ):
        """
        Get all the materials of building block.
        :return:
        """
        url_params = {}
        url_data_type = "/geometry/userroutes/materials"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
