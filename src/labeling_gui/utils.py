from pathlib import Path
import tifffile as tiff
import numpy as np
import os


def load_stack(filepath: str) -> np.ndarray:
    """
    Load a TIFF stack, take 5% (every 20th Bscan) and normalize them to [0, 1] datatype float32.
    """
    stack = tiff.imread(filepath).astype(np.float32)
    stack = (stack - np.min(stack)) / (np.max(stack) - np.min(stack))
    return stack[::20] # every 20th Bscan
#

def save_pngs(stack: np.ndarray, masks: np.ndarray, original_filename: str, output_directory: str) -> None:
    """
    Save selected slices of the stack and its corresponding mask as PNG files.
    """
    os.makedirs(output_directory, exist_ok=True)
    for i, (img, mask) in enumerate(zip(stack, masks)):
        img_filename = f"{original_filename}_Bscan_{i:03d}.png"
        mask_filename = f"{original_filename}_Bscan_{i:03d}_mask.png"
        tiff.imwrite(os.path.join(output_directory, img_filename), img)
        tiff.imwrite(os.path.join(output_directory, mask_filename), mask)
#
