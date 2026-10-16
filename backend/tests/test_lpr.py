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
    merge_plate_tokens,
    normalize_indian_plate,
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
    assert is_valid_plate_candidate("HONDA") is False  # badge text, no digits


def test_normalize_indian_plate():
    # Look-alike glyph fixes by position
    assert normalize_indian_plate("KAO1AB1234") == "KA01AB1234"
    assert normalize_indian_plate("TNO9BX7788") == "TN09BX7788"
    # Only Delhi has single-digit districts, so KA0S... must be KA05...
    assert normalize_indian_plate("KA0SMN0042") == "KA05MN0042"
    assert normalize_indian_plate("DL3CAF0021") == "DL3CAF0021"
    # BH series
    assert normalize_indian_plate("22BH1234AA") == "22BH1234AA"
    # Not plates
    assert normalize_indian_plate("HONDA") is None
    assert normalize_indian_plate("XUV500") is None


def test_merge_plate_tokens_joins_split_and_two_line_plates():
    box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    # Same row, split in two
    one_row = [(box(10, 10, 80, 40), "KA 01", 0.9), (box(90, 10, 200, 40), "AB 1234", 0.8)]
    assert ("KA01AB1234", 0.8) in merge_plate_tokens(one_row)
    # Two-line plate: second row sits directly under the first
    two_rows = [(box(40, 60, 160, 90), "AB1234", 0.95), (box(60, 10, 140, 40), "KA01", 0.9)]
    assert ("KA01AB1234", 0.9) in merge_plate_tokens(two_rows)
    # Far-apart text (badge above the bumper) is not glued onto the plate
    apart = [(box(60, 10, 140, 40), "HONDA", 0.9), (box(40, 300, 160, 330), "KA01AB1234", 0.9)]
    assert all(t != "HONDAKA01AB1234" for t, _ in merge_plate_tokens(apart))


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
