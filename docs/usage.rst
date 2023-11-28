Usage
=====

To use this library, you need to acquire authorization token and know the database API endpoint URL. 
Please contact responsible person from OWI-lab team to get them.

Once you have this information, you can use the library as follows:

.. code-block:: python
  
  from owimetadatabase_preprocessor.locations.io import LocationsAPI


  api_root = '<API endpoint URL>'
  head = {'Authorization': 'Token xxx'} # where xxx is the authorization token

  locations_api = LocationsAPI(api_root, header=head)  

  # Time to use methods from the API
  # E.g. to plot all asset locations (recommended to use in Jupyter notebook)
  locations_api.plot_assetlocations()


