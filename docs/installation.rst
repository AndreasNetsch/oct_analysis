Installation
============

Requirements
------------

* Python 3.6 or higher
* pip (Python package installer)

Basic Installation
------------------

You can install **oct_analysis** using pip:

.. code-block:: bash

    pip install oct_analysis

This will install the package and all required dependencies.

Development Installation
------------------------

For development, you can install **oct_analysis** from the source:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/yourusername/oct_analysis.git
       cd oct_analysis

2. Create and activate a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package in development mode with all dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

This will install the package in development mode, along with all development dependencies.

Dependencies
------------

The following packages are required:

* numpy
* openencv-python
* scikit-image
* matplotlib
* tifffile

Optional dependencies for development:

* pytest
* flake8
* black
* build
* twine
* pre-commit
