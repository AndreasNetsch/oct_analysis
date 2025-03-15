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


def test_read_tiff_returns_numpy_array():
    """Test that read_tiff returns a numpy array when given a valid file"""
    # Mock cv2.imread to return a fake image
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("cv2.imread", return_value=fake_image):
            image = read_tiff("fake_file.tiff")
            assert isinstance(image, np.ndarray)
            assert image.shape == (100, 100, 3)


def test_read_tiff_error_on_none_image():
    """Test that ValueError is raised when cv2.imread returns None"""
    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("cv2.imread", return_value=None):
            with pytest.raises(ValueError):
                read_tiff("invalid_image.tiff")
