def main():
    from ebi_oct.napari_plugin import widgets
    from qtpy.QtWidgets import QScrollArea, QWidget, QVBoxLayout
    from magicgui.widgets import Container

    widget_list = [
        widgets.load_oct,
        widgets.median_filter,
        widgets.roi_filter_2D,
        widgets.locate_window,
        widgets.zero_out_window,
        widgets.locate_substratum,
        widgets.zero_out_substratum,
        widgets.binarize,
        widgets.remove_outliers,
        widgets.save
    ]

    all_widgets = Container(
        widgets=widget_list,
        layout='vertical',
        name='Image Processing Tools',
        visible=True,
        labels=False
    )

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container_widget = QWidget()
    layout = QVBoxLayout(container_widget)
    layout.addWidget(all_widgets.native)
    scroll.setWidget(container_widget)
    scroll.setMinimumWidth(400)

    return scroll
