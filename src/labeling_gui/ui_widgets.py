from magicgui import magicgui
from magicgui.widgets import FileEdit
from qtpy.QtWidgets import QFileDialog, QPushButton
import napari
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info
import numpy as np
from pathlib import Path


from src.labeling_gui.utils import load_stack, save_pngs
from src.processing_functions import image_processing as ip


def create_gui(viewer: napari.Viewer) -> None:
    original_stack = {}
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
        original_stack['filename'] = Path(filepath).stem
        original_stack['directory'] = Path(filepath).parent
        original_stack['data'] = stack
        viewer.layers.clear()
        viewer.add_image(stack, name="original", colormap="gray")
        show_info(f"Loaded {filepath} with shape {stack.shape}. (Every 20th Bscan)")
    #

    @magicgui(call_button="Process")
    def process():
        nonlocal mask_stack
        mask_stack = np.zeros_like(original_stack['data'], dtype=np.uint8)
    #

    @magicgui(call_button="Save as PNGs",
              output_directory={"label": "Choose directory", "widget_type": FileEdit, "mode": "d"})
    def save_widget(output_directory) -> None:
        save_pngs(original_stack['data'], mask_stack, original_stack["filename"], str(output_directory))
        show_info(f"Saved PNGs to {output_directory}.")
    #

    open_button = QPushButton("Open TIF stack")
    open_button.clicked.connect(load_widget)

    viewer.window.add_dock_widget(open_button, area='right')
    viewer.window.add_dock_widget(process, area='right')
    viewer.window.add_dock_widget(save_widget, area='right')
#
