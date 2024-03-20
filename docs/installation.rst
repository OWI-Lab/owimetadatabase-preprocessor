Installation
============

Installation (recommended)
--------------------------

In your desired virtual environment with Python 3 and pip installed:

.. code-block:: bash
   
  pip install owimetadatabase-preprocessor


Installation (alternative)
--------------------------

In your desired virtual environment and directory with Python 3 and pip installed:

.. code-block:: bash

  git clone <repo-github-address>

  pip install <repo-local-name>


Installation (beta)
--------------------------

In case you want to install the latest beta version (note it may be outdated by the stable version):

.. code-block:: bash
   
  pip install owimetadatabase-preprocessor --pre


Contribution (for developers)
-----------------------------

If you want to contribute to the development of the package, you can, in your desired virtual environment and directory with Python 3 and pip installed:

.. code-block:: bash

  git clone <repo-address>

  pip install -e <repo-name>/[dev]

This way, you will install all the required dependecies and the package itself in editable mode, i.e. all changes to it will be reflected immediately locally so it can be tested.

The repository also has ``.lock`` file if you use ``poetry``.