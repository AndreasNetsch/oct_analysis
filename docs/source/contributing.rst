Contributing
============

We welcome contributions to **oct_analysis**!

Development Setup
----------------

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/AndreasNetsch/oct_analysis.git
       cd oct_analysis

2. Create a virtual environment and install development dependencies:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install -e ".[dev]"

3. Set up the pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Code Style
----------

This project uses:

- **Black** for code formatting
- **Flake8** for linting

The pre-commit hooks will automatically format and lint your code before each commit.

Testing
-------

Make sure to write tests for any new functionality. Run the tests with pytest:

.. code-block:: bash

    pytest

Pull Request Process
-------------------

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests
5. Submit a pull request
6. Pull requst will be checked by admin prior to merge with main branch

When you're ready to release a new version:

1. Update the version number in ``setup.py``
2. Create a new version tag (the tag name must start with "v"):

   .. code-block:: bash

       git tag vx.y.z
       git push origin vx.y.z

3. The GitHub Actions workflow will automatically build and publish the package to PyPI.
