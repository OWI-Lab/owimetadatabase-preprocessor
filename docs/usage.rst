Getting started
===============

To use this library, you need to acquire authorization token and know the database API endpoint URL. \
Please contact responsible person from OWI-lab team to get them.

Once you have this information and you installed the package according to the instructions \
(preferably into your virtual environment) with appropriate Python version, e.g. in your command line/terminal:

.. code-block:: bash
   
  pip install owimetadatabase-preprocessor

Then, in your Python code, you can start using (import) the library as follows:

.. code-block:: python
  
  from owimetadatabase_preprocessor.locations.io import LocationsAPI

This command will import the suitable API class (LocationsAPI here for the data about locations, see reference for others). \
Now, for convenience, you can store the API endpoint URL and authorization token in some variables:

.. code-block:: python

  api_root = '<API endpoint URL>'
  head = {'Authorization': 'Token xxx'}  #  where xxx is the authorization token

, and create an instance of the API class with this information to later establish connection to the database:

.. code-block:: python

  locations_api = LocationsAPI(api_root, header=head)  

This class has several methods for convenient access/processing of the data from the database. \
E.g. to plot all asset locations (recommended to use in Jupyter notebook):

.. code-block:: python

  locations_api.plot_assetlocations()

And others (see reference) to operate on the information about locations/geometry.
