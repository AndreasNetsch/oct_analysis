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
    """Test that read_tiff returns the expected outputs with correct metadata"""
    fake_image = np.zeros((10, 20, 30), dtype=np.uint8)
    fake_description = "slices=10\nunit=mm\nspacing=1.0"

    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("tifffile.TiffFile") as mock_tif:
            # Setup mock for image data
            mock_tif.return_value.__enter__.return_value.asarray.return_value = fake_image
            
            # Setup mock for metadata
            mock_page = mock.Mock()
            mock_page.tags = {
                'ImageDescription': mock.Mock(value=fake_description),
                'ImageLength': mock.Mock(value=20),
                'ImageWidth': mock.Mock(value=30),
                'XResolution': mock.Mock(value=(1,1)),
                'YResolution': mock.Mock(value=(1,1))
            }
            mock_tif.return_value.__enter__.return_value.pages = [mock_page]
            
            # Setup mock for series info
            mock_series = mock.Mock()
            mock_series.shape = (10, 20, 30)
            mock_series.dtype = np.uint8
            mock_series.axes = 'ZYX'
            mock_tif.return_value.__enter__.return_value.series = [mock_series]

            img, filename, metadata = read_tiff("fake_file.tiff")

            # Test image output
            assert isinstance(img, np.ndarray)
            assert img.shape == (10, 20, 30)
            assert img.dtype == np.uint8

            # Test filename output
            assert isinstance(filename, str)
            assert filename == "fake_file"

            # Test metadata output
            assert isinstance(metadata, dict)
            assert metadata['Z'] == 10
            assert metadata['Y'] == 20
            assert metadata['X'] == 30
            assert metadata['shape'] == (10, 20, 30)
            assert metadata['dtype'] == np.uint8.__name__
            assert metadata['axes'] == 'ZYX'
            assert metadata['XResolution'] == (1, 1)
            assert metadata['YResolution'] == (1, 1)
            assert metadata['unit'] == 'mm'
            assert metadata['spacing'] == 1.0


def test_read_tiff_error_on_none_image():
    """Test that ValueError is raised when image reading fails"""
    with mock.patch("os.path.isfile", return_value=True):
        with mock.patch("tifffile.TiffFile") as mock_tif:
            mock_tif.return_value.__enter__.return_value.asarray.return_value = None
            with pytest.raises(ValueError):
                read_tiff("invalid_image.tiff")
