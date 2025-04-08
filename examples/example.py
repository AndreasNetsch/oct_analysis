import oct_analysis.image_processing as oct
import glob
import os

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
    img = oct.untilt(img, thres=1, y_offset=0, top_crop=30) # Remove black area beneath substratum
    oct.save_tiff(img, output_folder, filename, metadata=metadata)

    #img_binary = oct.binary_mask(img, thresholding_method='yen', contrast=0.35, blurred=True, blur_size=1, outliers_size=3)
    #oct.save_tiff(img_binary, output_folder, f"{filename}_binary", metadata=metadata)