"""
Image processing functions for oct_analysis
"""

import cv2
import os


def read_tiff(file_path):
    """
    Read an image from a TIFF file.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file

    Returns
    -------
    numpy.ndarray
        The image as a numpy array

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file could not be read as an image
    """
    # Check if the file exists before trying to read it
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Use OpenCV to read the TIFF file
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to read image from {file_path}")

        return img
    except Exception as e:
        raise ValueError(f"Error reading TIFF file: {str(e)}")
