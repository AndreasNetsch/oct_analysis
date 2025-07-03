Examples
========

Reading TIFF Images
-------------------

To read a TIFF image using **oct_analysis**, you can use the ``read_tiff`` function:

.. code-block:: python

    from oct_analysis import read_tiff

    # Read a TIFF image
    image = read_tiff('path/to/your/image.tiff')

    # Print the shape of the image
    print(f"Image shape: {image.shape}")

Finding Substratum in Images
----------------------------

To find and process the substratum in an image, you can use the ``find_substratum`` function:

.. code-block:: python

    from oct_analysis import find_substratum

    # Find substratum in an image
    processed_image = find_substratum(
        img=image,
        start_x=0,                          # Starting x-coordinate
        y_max=image.shape[0]//2,            # Maximum y-coordinate to search
        roi_width=20,                       # Width of the region of interest
        scan_height=10,                     # Height of the scan area
        step_width=5                        # Width of the box to scan
    )

    # The function returns a processed image where areas above the detected substratum are set to black

Error Handling
--------------

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
