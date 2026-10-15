import sys
from pathlib import Path

# Add project root to sys.path so direct script execution succeeds
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pytest

from backend.app.lpr import (
    LicensePlateRecognizer,
    clean_plate_text,
    is_valid_plate_candidate,
    preprocess_plate_image,
)


def test_clean_plate_text():
    assert clean_plate_text("DL 01 AB 1234") == "DL01AB1234"
    assert clean_plate_text("ca-123-xyz!") == "CA123XYZ"
    assert clean_plate_text("  abc_999  ") == "ABC999"


def test_is_valid_plate_candidate():
    assert is_valid_plate_candidate("DL01AB1234") is True
    assert is_valid_plate_candidate("XYZ123") is True
    assert is_valid_plate_candidate("A1") is False  # too short (< 4)
    assert is_valid_plate_candidate("TOOLONGPLATE1234567") is False  # too long (> 12)
    assert is_valid_plate_candidate("----") is False


def test_preprocess_plate_image():
    # Synthetic small plate image
    img = np.full((30, 80, 3), 200, dtype=np.uint8)
    enhanced = preprocess_plate_image(img)
    assert enhanced is not None
    # Check that it converted to single channel grayscale
    assert len(enhanced.shape) == 2
    # Check that it upscaled
    assert enhanced.shape[0] >= 60
    assert enhanced.shape[1] >= 160


def test_lpr_synthetic_plate():
    """Create a synthetic license plate image and verify OCR reading."""
    plate_img = np.full((100, 350, 3), 245, dtype=np.uint8)
    # Draw dark blue / black text on white plate
    cv2.putText(
        plate_img,
        "DL01AB1234",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (10, 10, 10),
        4,
        cv2.LINE_AA,
    )

    recognizer = LicensePlateRecognizer(gpu=False)
    result = recognizer.read_plate(plate_img, min_confidence=0.20)
    assert result is not None
    plate_text, confidence = result
    assert "DL01AB1234" in plate_text or "1234" in plate_text
    assert confidence > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
