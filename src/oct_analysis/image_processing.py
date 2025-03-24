"""
Image processing functions for oct_analysis
"""

import cv2
import os
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


def find_substratum (img, start_x, y_max, roi_width, scan_height, box_width):
    """
    Find the substratum in an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array
    start_x : int
        The x-coordinate of the starting point
    y_max : int
        The maximum y-coordinate
    roi_width : int
        The width of the region of interest
    scan_height : int
        The height of the scan area
    box_width : int
        The width of the box to scan

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
    img = img[:, ::-1, :]
    slices, h, w = img.shape
    
    for slice_idx, img in enumerate(img):
        maxSum = 0
        memBot = 0
        # Find the bottom of the membrane in the first slice
        for i in range(y_max):
            roi = img[i, start_x:start_x+roi_width]
            sum_val = np.mean(roi)
            if sum_val > maxSum:
                maxSum = sum_val
                memBot = i
            memBot1 = memBot

    # Process each slice
        for x in range(start_x, w, box_width):
            memBot = memBot1
            for y in range(memBot - scan_height, memBot + scan_height, 1):
                # Ensure 'y' is within bounds for the image height
                if y < 0:
                    y = 0
                elif y >= img.shape[0]:  # img.shape[0] is the height of the image
                    y = img.shape[0] - 1
                roi = img[y, x:x+roi_width]
                sum_val = np.mean(roi)
                if sum_val > maxSum:
                    maxSum = sum_val
                    memBot = y
                if start_x == 0:
                    memBot1 = memBot
            maxSum = 0
            if memBot > 0:
                img[:memBot, x:x+box_width] = 0  # Set area to black
        #print(slice_idx)
    img = img[:, ::-1, :]
    return img