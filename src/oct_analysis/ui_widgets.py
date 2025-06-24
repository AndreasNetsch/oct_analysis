from magicgui import magicgui
from magicgui.widgets import Container
from magicgui.widgets import PushButton
from magicgui.widgets import Label
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QWidget, QScrollArea, QVBoxLayout
import napari
from napari.utils.notifications import show_info
import numpy as np
from pathlib import Path
import scipy.ndimage
from skimage import morphology

from src.oct_analysis import processing_functions as ip


def create_gui(viewer: napari.Viewer) -> None:
    """
    Create a GUI for loading and processing image stacks in Napari.
    """

    stack_container = {} # holds all relevant data during the session
    def update_widgets():
        locate_window_widget.ymax.value = (
            int(stack_container["original_normalized"].shape[1] * 0.25)
            if 'original_normalized' in stack_container
            else 100
        )
        locate_substratum_widget.ymin.value = (
            int(stack_container["original_normalized"].shape[1] * 0.6)
            if 'original_normalized' in stack_container
            else 200
        )
        locate_substratum_widget.ymax.value = (
            int(stack_container["original_normalized"].shape[1])
            if 'original_normalized' in stack_container
            else 1000
        )
        save_pngs_widget.output_directory.value = (
            str(stack_container['directory'])
            if 'directory' in stack_container
            else str(Path.home())
        )

    def load_widget():
        # hidden filedialog
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            'Select TIF file',
            '',
            'TIF files (*.tif *.tiff)'
        ) # type: ignore
        if not filepath:
            show_info("No file selected.")
            return # Abbrechen
        original_data, stack_container['filename'], _ = ip.read_tiff(filepath)
        stack_container['original_normalized'] = ip._normalize_tiff(original_data)
        stack_container['directory'] = Path(filepath).parent
        viewer.layers.clear()
        viewer.add_image(
            stack_container["original_normalized"],
            name="original_normalized",
            colormap="magma", gamma=2.0,
            visible=True
        )
        show_info(f"Loaded {stack_container['filename']}")

        update_widgets()

    @magicgui(
            call_button="Apply Median Filter",
            filter_size={"widget_type": "SpinBox", "min": 1, "max": 101, "step": 2, "value": 3}
    )
    def median_blur_widget(filter_size):
        stack_container['median_filtered'] = ip.median_blur(stack_container['original_normalized'], filter_size)
        if 'median_filtered' in viewer.layers:
            viewer.layers['median_filtered'].data = stack_container['median_filtered']
        else:
            viewer.add_image(stack_container['median_filtered'], name="median_filtered", colormap="magma", gamma=2.0, visible=False)

    @magicgui(call_button="Apply ROI Filter",
                roi_width={"widget_type": "SpinBox", "min": 1, "max": 101, "step": 2, "value": 11}
    )
    def roi_filter_widget(roi_width):
        stack_container['roi_filtered'] = scipy.ndimage.uniform_filter1d(stack_container["median_filtered"], size=roi_width, axis=2, mode='reflect')
        if 'roi_filtered' in viewer.layers:
            viewer.layers['roi_filtered'].data = stack_container['roi_filtered']
        else:
            viewer.add_image(stack_container['roi_filtered'], name="roi_filtered", colormap="magma", gamma=2.0, visible=False)

    @magicgui(call_button="Locate Window",
                ymin={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 0},
                ymax={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 100},
                y_offset={"widget_type": "SpinBox", "min": -100, "max": 100, "step": 1, "value": 0}
    )
    def locate_window_widget(ymin, ymax, y_offset):
        y_coords = np.argmax(stack_container['roi_filtered'][:, ymin:ymax, :], axis=1)
        y_coords += y_offset
        stack_container['window_coords'] = y_coords

        # vizualize y_coords as labels:
        z, x = stack_container['window_coords'].shape
        y = stack_container["median_filtered"].shape[1]
        label_stack = np.zeros((z, y, x), dtype=np.uint8)
        z_idx, x_idx = np.meshgrid(np.arange(z), np.arange(x), indexing='ij')
        y_idx = stack_container['window_coords']
        mask = (y_idx >= 0) & (y_idx < y)  # ensure indices are within bounds
        label_stack[z_idx[mask], y_idx[mask], x_idx[mask]] = 1

        if 'window_coords' in viewer.layers:
            viewer.layers['window_coords'].data = label_stack
        else:
            viewer.add_labels(label_stack, name="window_coords", visible=True)

    @magicgui(call_button="Zero Out Window")
    def zero_out_window_widget():
        stack_container['no_window'] = ip.zero_out_window(stack_container['original_normalized'], stack_container['window_coords'])
        if 'no_window' in viewer.layers:
            viewer.layers['no_window'].data = stack_container['no_window']
        else:
            viewer.add_image(stack_container['no_window'], name="no_window", colormap="magma", gamma=2.0, visible=True)

    @magicgui(call_button="Locate Substratum",
                ymin={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 200},
                ymax={"widget_type": "SpinBox", "min": 0, "max": 1000, "step": 1, "value": 1000},
                y_offset={"widget_type": "SpinBox", "min": -100, "max": 100, "step": 1, "value": 0}
    )
    def locate_substratum_widget(ymin, ymax, y_offset):
        y_coords = np.argmax(stack_container['roi_filtered'][:, ymin:ymax, :], axis=1)
        y_coords = y_coords + ymin + y_offset
        stack_container['substratum_coords'] = y_coords

        # vizualize y_coords as labels:
        z, x = stack_container['substratum_coords'].shape
        y = stack_container["median_filtered"].shape[1]
        label_stack = np.zeros((z, y, x), dtype=np.uint8)
        z_idx, x_idx = np.meshgrid(np.arange(z), np.arange(x), indexing='ij')
        y_idx = stack_container['substratum_coords']
        mask = (y_idx >= 0) & (y_idx < y) # ensure indices are within bounds
        label_stack[z_idx[mask], y_idx[mask], x_idx[mask]] = 2

        if 'substratum_coords' in viewer.layers:
            viewer.layers['substratum_coords'].data = label_stack
        else:
            viewer.add_labels(label_stack, name="substratum_coords", visible=True)

    @magicgui(call_button="Zero Out Substratum")
    def zero_out_substratum_widget():
        stack_container['no_substratum'] = ip.zero_out_substratum(stack_container["no_window"] if 'no_window' in stack_container else stack_container['original_normalized'], stack_container['substratum_coords'])
        if 'no_substratum' in viewer.layers:
            viewer.layers['no_substratum'].data = stack_container['no_substratum']
        else:
            viewer.add_image(stack_container['no_substratum'], name="no_substratum", colormap="magma", gamma=2.0, visible=True)

    @magicgui(call_button="Binarize")
    def binarize_widget():
        stack_container['binary'] = ip.binarize_ignore_zeros(img=stack_container['no_substratum'])
        stack_container["binary"] = stack_container["binary"].astype(np.uint8)
        if 'binary' in viewer.layers:
            viewer.layers['binary'].data = stack_container['binary']
        else:
            viewer.add_labels(stack_container['binary'], name="binary", visible=True)

    @magicgui(call_button="Remove Outliers",
                outliers_size={"widget_type": "SpinBox", "min": 1, "max": 20, "step": 1, "value": 2}
            )
    def remove_outliers_widget(outliers_size):
        stack_container["binary"] = morphology.remove_small_objects(stack_container["binary"].astype(bool), min_size=outliers_size, connectivity=1)
        stack_container["binary"] = stack_container["binary"].astype(np.uint8)
        viewer.layers['binary'].data = stack_container['binary']

    @magicgui(call_button="Save Processed Stack",
                output_directory={"widget_type": "FileEdit", "mode": "d", "value": str(Path.home())}
            )
    def save_pngs_widget(output_directory):
        ip.save_pngs(
            original_stack=stack_container['reduced_data'] if 'reduced_data' in stack_container else stack_container['original_normalized'],
            binary_stack=stack_container['binary'],
            original_filename= stack_container['filename'],
            output_directory=output_directory
        )
        show_info(f"Saved images and masks to {output_directory}.")


    # Place all widgets in the napari-UI
    open_button = PushButton(label="Open TIF stack")
    open_button.clicked.connect(load_widget)

    custom_widgets = [
        open_button,
        Label(value='<b>--------------Preprocessing--------------</b>'),
        median_blur_widget,
        roi_filter_widget,
        Label(value='<b>--------------Remove Optical Window--------------</b>'),
        locate_window_widget,
        zero_out_window_widget,
        Label(value='<b>--------------Remove Substratum--------------</b>'),
        locate_substratum_widget,
        zero_out_substratum_widget,
        Label(value='<b>--------------Binarization--------------</b>'),
        binarize_widget,
        remove_outliers_widget,
        Label(value='<b>--------------Save as training data--------------</b>'),
        save_pngs_widget
    ]

    all_widgets = Container(widgets=custom_widgets,
                            layout='vertical',
                            name='Image Processing Tools',
                            visible=True,
                            labels=False)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container_widget = QWidget()
    layout = QVBoxLayout(container_widget)
    layout.addWidget(all_widgets.native)
    scroll.setWidget(container_widget)
    scroll.setMinimumWidth(400)

    viewer.window.add_dock_widget(scroll, area='right', name='Image Processing Tools')
