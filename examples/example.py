from oct_analysis.image_processing import read_tiff


if __name__ == "__main__":
    # Read an image from a TIFF file
    img = read_tiff("data/image.tiff")
    print(img)