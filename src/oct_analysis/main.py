import napari

from src.oct_analysis.ui_widgets import create_gui


if __name__ == "__main__":
    viewer = napari.Viewer()
    create_gui(viewer)
    napari.run()
