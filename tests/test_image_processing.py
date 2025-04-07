"""
Tests for the image_processing module
"""

import pytest
import numpy as np
from unittest import mock

from oct_analysis.image_processing import read_tiff


def test_read_tiff_file_not_found():
    """Test that FileNotFoundError is raised when the file doesn't exist"""
    with mock.patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError):
            read_tiff("nonexistent_file.tiff")


def test_read_tiff_returns_expected_outputs():
    fake_image = np.zeros((10, 20, 30), dtype=np.uint8)
    fake_metadata = {'XResolution': (1, 1), 'YResolution': (1, 1)}

    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("tifffile.TiffFile") as mock_tif:
            mock_tif.return_value.__enter__.return_value.asarray.return_value = fake_image
            mock_tif.return_value.__enter__.return_value.pages[0].tags = {
                'XResolution': mock.Mock(value=(1,1)),
                'YResolution': mock.Mock(value=(1,1))
            }
            mock_tif.return_value.__enter__.return_value.series = [mock.Mock(axes='ZYX')]

            img, filename, metadata = read_tiff("fake_file.tiff")

            assert isinstance(img, np.ndarray)
            assert img.shape == (10, 20, 30)
            assert isinstance(filename, str)
            assert isinstance(metadata, dict)


def test_read_tiff_error_on_none_image():
    """Test that ValueError is raised when cv2.imread returns None"""
    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("tifffile.imread", return_value=None):
            with pytest.raises(ValueError):
                read_tiff("invalid_image.tiff")
