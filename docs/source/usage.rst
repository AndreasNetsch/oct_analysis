Usage
=====

Reading TIFF Images
------------------

To read a TIFF image using **oct_analysis**, you can use the ``read_tiff`` function:

.. code-block:: python

    from oct_analysis import read_tiff

    # Read a TIFF image
    image = read_tiff('path/to/your/image.tiff')

    # Print the shape of the image
    print(f"Image shape: {image.shape}")

Error Handling
-------------

The ``read_tiff`` function provides built-in error handling:

- If the file doesn't exist, a ``FileNotFoundError`` is raised
- If the file can't be read as an image, a ``ValueError`` is raised

Example with error handling:

.. code-block:: python

    from oct_analysis import read_tiff

    try:
        image = read_tiff('path/to/your/image.tiff')
        print(f"Image shape: {image.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
