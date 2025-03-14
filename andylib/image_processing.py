"""
Image processing functions for andylib
"""

import cv2
import numpy as np


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
    try:
        # Use OpenCV to read the TIFF file
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to read image from {file_path}")
            
        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading TIFF file: {str(e)}") 