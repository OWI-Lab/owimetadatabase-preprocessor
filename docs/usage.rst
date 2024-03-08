Getting started
===============

To use this library, you need to acquire authorization token. \
Please contact responsible person from OWI-lab team to get them.

Once you have this information and you installed the package according to the instructions \
(preferably into your virtual environment) with appropriate Python version, e.g. in your command line/terminal:

.. code-block:: bash
   
  pip install owimetadatabase-preprocessor

Then, in your Python code, you can start using (import) the library as follows:

.. code-block:: python
  
  from owimetadatabase_preprocessor.locations.io import LocationsAPI

This command will import the suitable API class (LocationsAPI here for the data about locations, see reference for others). \
Now, for convenience, you can store your authorization token in a variable:

.. code-block:: python

  # Not recommended as it increases risks of leaks!!!
  TOKEN = <your-token-string>

  # One of the recommended ways, using environment variables
  # E.g. with dotenv package:
  # pip install python-dotenv==1.0.0
  # Afterwards, define your OWIMETADB_TOKEN env variable (preffered in .env file, no source control!!!)
  from dotenv import load_dotenv
  load_dotenv()
  TOKEN = os.getenv('OWIMETADB_TOKEN')

, and create an instance of the API class with this information to later establish connection to the database:

.. code-block:: python

  locations_api = LocationsAPI(token=TOKEN)

Please note, that you can specify the API endpoint URL as well, if it is different from the default one. \

This class has several methods for convenient access/processing of the data from the database. \
E.g. to plot all asset locations (recommended to use in Jupyter notebook):

.. code-block:: python

  locations_api.plot_assetlocations()

And others (see reference) to operate on the information about locations/geometry. For examples of the usage, \
please check `this notebook <https://github.com/OWI-Lab/owimetadatabase-preprocessor/blob/main/examples/basic_example.ipynb>`_.