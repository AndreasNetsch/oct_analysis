"""
Image processing functions for oct_analysis
"""

import cv2
import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk





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

def select_tiff_folder():
    """
    Opens a window to select a folder containing TIFF files.

    Returns
    -------
    str
        The path to the selected folder.
    """
    root = ctk.CTk()
    root.withdraw()  # Hide the root window
    folder_path = ctk.filedialog.askdirectory(title="Select Folder")
    if folder_path:
        print(f"Selected Folder: {folder_path}")
    else:
        print("No folder selected.")
    return folder_path

# Pre-Processing functions
def convert_to_8bit(img):
    """
    Converts an image to 8-bit format.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.

    Returns
    -------
    numpy.ndarray
        The image as a numpy array.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
    else:
        # If the image is integer-based (like uint32), just scale it to [0, 255]
        img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
    return img

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

def find_max_zero (img, top_crop):

    """
    Finds the maximum number of zero pixels in any column and removes the top rows according to the maximum number of zero pixels.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.
    top_crop : int
        The number of pixels to crop from the top of the image. (default = 0)

    Returns
    -------
    numpy.ndarray
        The image as a numpy array.
    """

    slices, h, w = img.shape
    # Variable to keep track of the maximum number of zero pixels in any column
    max_zero_pixels = 0

    # Loop through each slice
    for s in range(slices):
        # Get the current slice
        slice_img = img[s]

        # Loop through each pixel column (x) in the slice
        for x in range(w):
            profile = slice_img[:, x]

            # Count the number of zero pixels in the column
            zero_count = np.sum(profile == 0)

            # Update the maximum zero pixel count if necessary
            max_zero_pixels = max(max_zero_pixels, zero_count)
    # Flip the stack vertically
    img = img[:, ::-1, :]
    # Remove the top rows according to max_zero_pixels
    img = img[:, max_zero_pixels+top_crop:, :]
    # Flip the stack vertically
    img = img[:, ::-1, :]
    return img

def untilt(img, thres, y_offset, top_crop):

    """
    Finds the substratum in an image and tilts it until the substratum is horizontal.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.
    thres : int
        The threshold for the substratum.
    y_offset : int
        The y-offset of the substratum.
    top_crop : int
        The number of pixels to crop from the top of the image.

    Returns
    -------
    numpy.ndarray
        The image as a numpy array.
    """

    slices, h, w = img.shape
    img = img[:, ::-1, :]
    # Create an empty array for the resulting image
    new_img_array = np.zeros_like(img)

    for s in range(slices):
        # Get the current slice
        slice_img = img[s]

        # Loop through each pixel column (x) in the slice
        for x in range(w):
            profile = slice_img[:, x]
        
            # Extract non-zero pixels from the column
            non_zero_pixels = profile[profile > 0]

            if len(non_zero_pixels) > thres:
                # Move non-zero pixels to the top of the column
                new_img_array[s, :len(non_zero_pixels), x] = non_zero_pixels

    img = new_img_array[:, ::-1, :]
    # Remove the bottom rows after processing all slices, according to y_offset
    img = img[:, :-y_offset, :]
    img = img[:, ::-1, :]
    img = find_max_zero(img, top_crop)
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

def generate_Height_Map(img, voxel_size):

    """
    Generates a height map from an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image as a numpy array.
    voxel_size : tuple
        The voxel size of the image. (z, y, x) in mm
    filename : str
        The filename of the image.

    Returns
    -------  
    height_map : numpy.ndarray
        The height map as a numpy array.
    min_thickness : float
        The minimum thickness of the image.
    mean_thickness : float
        The mean thickness of the image.
    max_thickness : float 
        The maximum thickness of the image.
    std_thickness : float
        The standard deviation of the thickness of the image.
    surface_coverage_3px : float
        The surface coverage of the image ignoring the bottom 3 pixels.
    surface_coverage_5px : float
        The surface coverage of the image ignoring the bottom 5 pixels.
    surface_coverage_10px : float
        The surface coverage of the image ignoring the bottom 10 pixels.
    Raises
    ------
    No Errors   
    """
    # Change to 32-bit float for calculations
    img = img.astype(np.float32)
    #image_stack=np.flip(image_stack, axis=1)

    # Normalize by dividing by 255
    img /= 255.0
    
    # Get voxel size (assuming isotropic pixels in X and Y, different Z)
    slice_thickness, dy, dx = voxel_size  # Adjust if metadata is available
    print(f"Slice Thickness = {slice_thickness} mm")
    print(f"Pixel_dim_y = {dy} mm")
    print(f"Pixel_dim_x = {dx} mm")
    
    resliced_stack = np.transpose(img, (1, 2, 0))
    #filename = str(filename) + "_height_map.tif"
    #tiff.imwrite(filename, resliced_stack.astype(np.float32))
    # Flip along the new z-axis to correct orientation
    #resliced_stack = np.flip(resliced_stack, axis=0) 
    slices, h, w = resliced_stack.shape
    # Calculate the height map (maximum z value for each (x, y) position)
    max_indices = np.argmax(resliced_stack, axis=0)  # Maximum index along z-axis for each (x, y)  
    height_map = (slices-max_indices) 

    # Ensure zero indices remain zero
    height_map[max_indices == 0] = 0 # Set zero indices to zero height
    height_map = height_map * slice_thickness  # Convert index to physical height

    # Generate and save Fire-coded height map
    #filename = filename.replace("_height_map.tif", "_fire_map.png")
    plt.imshow(height_map, cmap='inferno')
    plt.axis('off')
    plt.colorbar(label='Height (mm)')
    plt.savefig("height_map.tif", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Compute thickness statistics
    valid_pixels = height_map
    min_thickness = np.min(valid_pixels) if valid_pixels.size > 0 else 0
    mean_thickness = np.mean(valid_pixels) if valid_pixels.size > 0 else 0
    max_thickness = np.max(valid_pixels) if valid_pixels.size > 0 else 0
    std_thickness = np.std(valid_pixels) if valid_pixels.size > 0 else 0
    
    # Compute surface coverage using histogram method
    histogram, _ = np.histogram(height_map, bins=256, range=(0, np.max(height_map)))
    n_pixels = height_map.size
    surface_coverage_3px = (1 - np.sum(histogram[:4]) / n_pixels) * 100
    surface_coverage_5px = (1 - np.sum(histogram[:6]) / n_pixels) * 100
    surface_coverage_10px = (1 - np.sum(histogram[:11]) / n_pixels) * 100
    
    # Prepare results
    results = [
        "Statistics:",
        "-------------",
        f"Slice Thickness = {slice_thickness} mm",
        f"Min Thickness = {min_thickness} mm",
        f"Mean Thickness = {mean_thickness:.2f} mm",
        f"Max Thickness = {max_thickness} mm",
        f"Standard Deviation of Thickness = {std_thickness:.2f} mm",
        f"Surface Coverage 3px = {surface_coverage_3px:.2f} %",
        f"Surface Coverage 5px = {surface_coverage_5px:.2f} %",
        f"Surface Coverage 10px = {surface_coverage_10px:.2f} %"
    ]
    # Print results
    for line in results:
        print(line)
    
    return height_map, min_thickness, mean_thickness, max_thickness, std_thickness, surface_coverage_3px, surface_coverage_5px, surface_coverage_10px