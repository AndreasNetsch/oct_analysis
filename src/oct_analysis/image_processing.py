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


# Pre-Processing functions

def find_substratum(img, start_x, y_max, roi_width, scan_height, step_width):

    """
    Find the substratum in an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.
    start_x : int
        The x-coordinate of the starting point of the scan (default = 0)
    y_max : int
        Maximum y-coordinate of the substratum at the starting point (typically the half of the height of the image)
    roi_width : int
        Width of the region of interest (default = 20)
    scan_height : int
        Height of the scan area (default = 10)
    step_width : int
        Width of the scan steps (default = 5)

    Returns
    -------
    numpy.ndarray
        The image as a numpy array

    Raises
    ------
    No Errors
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
        for x in range(start_x, w, step_width):
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
                img[:memBot, x:x+step_width] = 0  # Set area to black
        #print(slice_idx)
    img = img[:, ::-1, :]
    return img


# Post-Processing functions
def voxel_count(img, voxel_size):

    """
    Counts the number of white pixels in the image and calculates the volume.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.
    voxel_size : tuple
        The voxel size of the image. (z, y, x) in mm

    Returns
    -------
    volume : float
        The volume of the image in mm³
    """

    no_of_white_pixels = np.sum(img == 255)
    volume = no_of_white_pixels * voxel_size[0] * voxel_size[1] * voxel_size[2]
    
    # Prepare result string
    result_text = (
        f"Volume = {volume} mm³\n"
        f"Pixel_dim_z = {voxel_size[0]} mm\n"
        f"Pixel_dim_y = {voxel_size[1]} mm\n"
        f"Pixel_dim_x = {voxel_size[2]} mm\n"
    )

    print(result_text)  # Print results to console
    return volume
