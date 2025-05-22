

from src.oct_analysis.image_processing import find_substratum
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


def main():
    root = tk.Tk()
    app = ImageGUI(root)
    root.mainloop()
###


class ImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling GUI")
        self.image_path = None
        self.image_array = None

        # Frames for layout
        top_frame = tk.Frame(root)
        top_frame.pack()
        bottom_frame = tk.Frame(root)
        bottom_frame.pack()

        # Buttons
        tk.Button(top_frame, text='Open Image', command=self.open_image).grid(row=0, column=0)
        tk.Button(top_frame, text='Apply Functions', command=self.apply_functions).grid(row=0, column=1)

        # Checkboxes for functions
        self.use_substratum = tk.BooleanVar()
        self.use_window = tk.BooleanVar()
        tk.Checkbutton(top_frame, text='Find Substratum', variable=self.use_substratum).grid(row=1, column=0)
        tk.Checkbutton(top_frame, text='Find Window', variable=self.use_window).grid(row=1, column=1)

        # Parameters for functions
        tk.Label(top_frame, text='ROI width:').grid(row=2, column=0)
        self.roi_width_entry = tk.Entry(top_frame, width=5)
        self.roi_width_entry.insert(0, '10')
        self.roi_width_entry.grid(row=2, column=1)

        tk.Label(top_frame, text='Median Size:').grid(row=3, column=0)
        self.median_size_entry = tk.Entry(top_frame, width=5)
        self.median_size_entry.insert(0, '7')
        self.median_size_entry.grid(row=3, column=1)

        tk.Label(top_frame, text='Y Offset:').grid(row=4, column=0)
        self.y_offset_entry = tk.Entry(top_frame, width=5)
        self.y_offset_entry.insert(0, '4')
        self.y_offset_entry.grid(row=4, column=1)

        tk.Label(top_frame, text='Max Range %:').grid(row=5, column=0)
        self.max_range_entry = tk.Entry(top_frame, width=5)
        self.max_range_entry.insert(0, '0.5')
        self.max_range_entry.grid(row=5, column=1)

        # Image display area
        self.original_label = tk.Label(bottom_frame, text='Original Image')
        self.original_label.grid(row=0, column=0)
        self.processed_label = tk.Label(bottom_frame, text='Processed Image')
        self.processed_label.grid(row=0, column=1)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.tif")])
        image = Image.open(self.image_path)
        self.image_array = np.array(image)
        self.display_image(image, self.original_label)

    def apply_functions(self):
        if self.image_array is None:
            return
        
        img = self.image_array.copy()
        mask = np.zeros_like(img, dtype=np.uint8)
        roi_width = int(self.roi_width_entry.get())
        median_size = int(self.median_size_entry.get())
        y_offset = int(self.y_offset_entry.get())
        max_range_percent = float(self.max_range_entry.get())

        if self.use_substratum.get():
            mask, img = find_substratum(
                self.image_path,
                roi_width=roi_width,
                median_size=median_size,
                y_offset=y_offset,
                max_range_percent=max_range_percent
            )

        if self.use_window.get():
            mask, img = find_window(
                self.image_path,
                roi_width=roi_width,
                median_size=median_size,
                y_offset=y_offset,
                max_range_percent=max_range_percent
            )

        self.display_image(Image.fromarray(img), self.processed_label)

    def display_image(self, img, label):
        img = img.copy()
        img.thumbnail((800, 800))
        tk_img = ImageTk.PhotoImage(img)
        label.imgtk = tk_img  # Keep a reference to avoid garbage collection
        label.config(image=tk_img)
##

if __name__ == "__main__":
    main()
