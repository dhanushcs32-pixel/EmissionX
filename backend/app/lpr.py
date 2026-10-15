import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("emission_x.lpr")


def clean_plate_text(raw_text: str) -> str:
    """Removes non-alphanumeric characters and converts to uppercase."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
    return cleaned.upper()


def is_valid_plate_candidate(text: str) -> bool:
    """Validates if cleaned text resembles a valid license plate string."""
    # Plate length standard is typically between 4 and 12 characters
    if not (4 <= len(text) <= 12):
        return False
    # Most plates contain at least one digit and one letter
    has_digit = any(c.isdigit() for c in text)
    has_letter = any(c.isalpha() for c in text)
    return has_digit or has_letter


def preprocess_plate_image(image: np.ndarray) -> np.ndarray:
    """Applies enhancement filters to optimize OCR accuracy."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Upscale if the crop is small
    h, w = gray.shape[:2]
    if h < 60 or w < 160:
        scale = max(2.0, 160.0 / max(w, 1))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


class LicensePlateRecognizer:
    """EasyOCR-based Automatic License Plate Recognition engine."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True):
        import easyocr
        import torch

        use_gpu = gpu and torch.cuda.is_available()
        lang_list = languages or ["en"]
        logger.info(f"Initializing EasyOCR reader (languages={lang_list}, gpu={use_gpu})...")
        self.reader = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)

    def extract_lower_roi(self, vehicle_crop: np.ndarray, lower_ratio: float = 0.45) -> np.ndarray:
        """Extracts the lower portion of the vehicle crop where plates are commonly placed."""
        h, w = vehicle_crop.shape[:2]
        start_y = int(h * (1.0 - lower_ratio))
        return vehicle_crop[start_y:h, 0:w]

    def read_plate(self, image_crop: np.ndarray, min_confidence: float = 0.35) -> Optional[Tuple[str, float]]:
        """Performs OCR on an image crop and returns (plate_text, confidence)."""
        if image_crop is None or image_crop.size == 0:
            return None

        h, w = image_crop.shape[:2]
        if h < 20 or w < 40:
            return None

        preprocessed = preprocess_plate_image(image_crop)

        # Allow uppercase alphanumeric characters
        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            results = self.reader.readtext(
                preprocessed,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            logger.warning(f"OCR inference exception: {e}")
            return None

        candidates = []
        for bbox, text, conf in results:
            cleaned = clean_plate_text(text)
            if is_valid_plate_candidate(cleaned) and conf >= min_confidence:
                candidates.append((cleaned, float(conf)))

        if not candidates:
            return None

        # Return the candidate with the highest OCR confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def read_vehicle_plate(
        self,
        vehicle_crop: np.ndarray,
        min_confidence: float = 0.35,
    ) -> Optional[Tuple[str, float]]:
        """Attempts OCR first on the lower bumper ROI, then falls back to full vehicle crop."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # Try lower region first (typical bumper/plate zone)
        lower_roi = self.extract_lower_roi(vehicle_crop, lower_ratio=0.50)
        res = self.read_plate(lower_roi, min_confidence=min_confidence)
        if res:
            return res

        # Fallback to full crop
        return self.read_plate(vehicle_crop, min_confidence=min_confidence)
