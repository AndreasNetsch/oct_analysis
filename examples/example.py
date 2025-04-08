import oct_analysis.image_processing as oct
import glob
import os
import numpy as np

input_folder = oct.select_tiff_folder()
output_folder = oct.select_tiff_folder()
tiff_files = glob.glob(os.path.join(input_folder, '*.tif'))

# This example file is for the PVC example image

for input_filename in tiff_files:
    img, filename, metadata = oct.read_tiff(input_filename)
    img = oct.convert_to_8bit(img)
    slices, h, w =img.shape
     # Identify the optical window - find the maximum intensity in the region of interest
    img = oct.find_substratum(img, start_x=0, y_max=h//4, roi_width=20, scan_height=10, step_width=5)
    img = oct.find_max_zero(img, top_crop=0)

    # Identifies and removes the substratum - find the maximum intensity in the region of interest
    slices, h, w = img.shape
    img = oct.find_substratum(img, start_x=0, y_max=h, roi_width=20, scan_height=10, step_width=5)
    img = oct.untilt(img, thres=1, y_offset=7, top_crop=30) # Remove black area beneath substratum
    oct.save_tiff(img, output_folder, filename, metadata=metadata)

    # Create a binary mask of the image
    img_binary_raw = oct.binary_mask(img, thresholding_method='yen', contrast=0.35, blurred=False, blur_size=0, outliers_size=0)
    img_binary_blurred = oct.binary_mask(img, thresholding_method='yen', contrast=0.35, blurred=True, blur_size=5, outliers_size=5)
    img_binary_difference = np.clip(img_binary_blurred.astype(np.int16) - img_binary_raw.astype(np.int16), 0, 255).astype(np.uint8)
    img_binary = np.clip(img_binary_blurred.astype(np.int16) - img_binary_difference.astype(np.int16), 0, 255).astype(np.uint8)
    
    oct.save_tiff(img_binary, output_folder, f"{filename}_binary", metadata=metadata)


    # post-processing
    x_resolution = metadata.get('XResolution', 1.0)  # pixels per mm
    y_resolution = metadata.get('YResolution', 1.0)  # pixels per mm
    x_voxel_size = round((x_resolution[1]/x_resolution[0]), 4)   # mm/px     
    y_voxel_size = round((y_resolution[1]/y_resolution[0]), 4)   # mm/px
    z_voxel_size = round(metadata.get('spacing', 1.0), 4)      
    biovolume = oct.voxel_count(img_binary, voxel_size=(z_voxel_size, y_voxel_size, x_voxel_size))
    height_map, min_thickness, mean_thickness, max_thickness, std_thickness, substratum_coverage = oct.generate_Height_Map(img_binary, voxel_size=(z_voxel_size, y_voxel_size, x_voxel_size), filename=filename, output_folder=output_folder, vmin=0, vmax=0.5)
    B_map, min_thickness_B, mean_thickness_B_map, max_thickness_B_map, std_thickness_B_map, substratum_coverage_B_map = oct.generate_B_Map(img_binary, voxel_size=(z_voxel_size, y_voxel_size, x_voxel_size), filename=filename, output_folder=output_folder, vmin=0, vmax=0.5)