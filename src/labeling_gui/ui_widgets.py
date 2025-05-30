from magicgui import magicgui
from magicgui.widgets import FileEdit
from qtpy.QtWidgets import QFileDialog, QPushButton
import napari
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info
import numpy as np
from pathlib import Path
import scipy
import scipy.ndimage

from src.labeling_gui.utils import load_stack, save_pngs
from src.processing_functions import image_processing as ip


def create_gui(viewer: napari.Viewer) -> None:
    stack_container = {}
    # processed_stack = None
    mask_stack = None

    """
    Create a GUI for loading and processing image stacks in Napari.
    """

    def load_widget():
        # hidden filedialog
        filepath, _ = QFileDialog.getOpenFileName(
            None, 'Select TIF file', '', 'TIF files (*.tif *.tiff)')
        if not filepath:
            show_info("No file selected.")
            return # Abbrechen
        stack = load_stack(filepath)
        stack_container['filename'] = Path(filepath).stem
        stack_container['directory'] = Path(filepath).parent
        stack_container['original_data'] = stack
        viewer.layers.clear()
        viewer.add_image(stack, name="original", colormap="gray")
        show_info(f"Loaded {filepath} with shape {stack.shape}. (Every 20th Bscan)")
    #

    @magicgui(call_button="Apply Median Blur",
                filter_size={"widget_type": "SpinBox", "min": 1, "max": 17, "step": 2, "value": 3})
    def median_blur_widget(filter_size: int = 3) -> None:
        stack_container['median_filtered'] = scipy.ndimage.median_filter(stack_container['original_data'], size=(1, filter_size, filter_size))
        viewer.add_image(stack_container['median_filtered'], name="median_filtered", colormap="gray", visible=False)
    #

    @magicgui(call_button="Remove Window",
                roi_size={"widget_type": "SpinBox", "min": 1, "max": 21, "step": 2, "value": 11},
                y_offset={"widget_type": "SpinBox", "min": -10, "max": 10, "step": 1, "value": 0},
                ymin={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 0},
                ymax={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 1000})
    def remove_window_widget(roi_size: int = 11,
                                y_offset: int = 0,
                                ymin: int = 0,
                                ymax: int = 0) -> None:
        
        # 1. precompute roi-mean für ganzen stack:
        filtered = scipy.ndimage.uniform_filter1d(stack_container["median_filtered"], size=roi_size, axis=2, mode='reflect')
        # 2. maximum intensity in bestimmtem y-bereich finden:
        y_coords = np.argmax(filtered[:, ymin:ymax, :], axis=1)
        # apply y-offset
        y_coords += y_offset
        # 3. vizualize y_coords as labels:
        z, x = y_coords.shape
        y = stack_container["median_filtered"].shape[1]
        label_stack = np.zeros((z, y, x), dtype=np.uint8)
        z_idx, x_idx = np.meshgrid(np.arange(z), np.arange(x), indexing='ij')
        y_idx = y_coords
        mask = (y_idx >= 0) & (y_idx < y)  # ensure indices are within bounds
        label_stack[z_idx[mask], y_idx[mask], x_idx[mask]] = 1
        viewer.add_labels(label_stack, name="y_coords", visible=True)
    #

    @magicgui(call_button="Save", labels=False, 
              output_directory={"widget_type": FileEdit, "mode": "d"})
    def save_widget(output_directory) -> None:
        save_pngs(stack_container['original_data'], mask_stack, stack_container["filename"], str(output_directory))
        show_info(f"Saved PNGs to {output_directory}.")
    #


    open_button = QPushButton("Open TIF stack")
    open_button.clicked.connect(load_widget)

    viewer.window.add_dock_widget(open_button, area='right')
    viewer.window.add_dock_widget(median_blur_widget, area='right')
    viewer.window.add_dock_widget(remove_window_widget, area='right')
    viewer.window.add_dock_widget(save_widget, area='right')
#
