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
    # Plates contain both digits and letters; this rejects badges like "HONDA"
    has_digit = any(c.isdigit() for c in text)
    has_letter = any(c.isalpha() for c in text)
    return has_digit and has_letter


# OCR confusions between look-alike glyphs, applied by position in the plate
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "T": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}

# Indian layouts as (kind, length range) segments:
#   standard  KA 01 AB 1234  -> 2 letters, 1-2 digits, 0-3 letters, 4 digits
#   BH series 22 BH 1234 AA  -> 2 digits, "BH", 4 digits, 1-2 letters
INDIAN_LAYOUTS = [
    [("L", 2, 2), ("D", 1, 2), ("L", 0, 3), ("D", 4, 4)],
    [("D", 2, 2), ("BH", 2, 2), ("D", 4, 4), ("L", 1, 2)],
]


def _coerce(seg: str, kind: str) -> Optional[Tuple[str, int]]:
    """Force a segment to letters/digits; returns (text, substitutions) or None."""
    if kind == "BH":
        fixed = "".join(TO_LETTER.get(c, c) for c in seg)
        return (fixed, sum(a != b for a, b in zip(seg, fixed))) if fixed == "BH" else None
    out, subs = [], 0
    for c in seg:
        want_digit = kind == "D"
        if c.isdigit() == want_digit:
            out.append(c)
            continue
        m = (TO_DIGIT if want_digit else TO_LETTER).get(c)
        if m is None:
            return None
        out.append(m)
        subs += 1
    return "".join(out), subs


def normalize_indian_plate(text: str) -> Optional[str]:
    """Map raw OCR text onto an Indian plate layout, fixing O/0, I/1, S/5 style mix-ups.

    Returns the corrected plate, or None if the text cannot be an Indian plate.
    """
    best: Optional[Tuple[int, str]] = None

    def walk(layout, pos, acc, subs, lens):
        nonlocal best
        if not layout:
            # Only Delhi uses a single-digit district code (DL 3C ...); elsewhere it is padded (KA 05)
            standard = len(lens) == 4 and lens[0] == 2 and acc[:2].isalpha()
            if standard and lens[1] == 1 and acc[:2] != "DL":
                return
            if pos == len(text) and (best is None or subs < best[0]):
                best = (subs, acc)
            return
        kind, lo, hi = layout[0]
        for n in range(lo, hi + 1):
            if pos + n > len(text):
                break
            res = _coerce(text[pos:pos + n], kind)
            if res is not None:
                walk(layout[1:], pos + n, acc + res[0], subs + res[1], lens + [n])

    for layout in INDIAN_LAYOUTS:
        walk(layout, 0, "", 0, [])
    # Too many substitutions means we are inventing a plate, not correcting one
    if best is None or best[0] > max(2, len(text) // 3):
        return None
    return best[1]


def _box(bbox) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def merge_plate_tokens(results, max_tokens: int = 3) -> List[Tuple[str, float]]:
    """Builds plate candidates from EasyOCR tokens, joining pieces that sit together.

    EasyOCR often splits one plate into several boxes ("KA 01" / "AB 1234", or the two
    rows of a two-line plate). Each neighbouring run of up to max_tokens boxes, read
    left-to-right then top-to-bottom, becomes a candidate scored by its weakest token.
    """
    toks = []
    for bbox, text, conf in results:
        t = clean_plate_text(text)
        if t:
            toks.append((_box(bbox), t, float(conf)))
    if not toks:
        return []
    med_h = float(np.median([b[3] - b[1] for b, _, _ in toks])) or 1.0
    # Reading order: group into rows by vertical centre, then left to right
    toks.sort(key=lambda k: (round(((k[0][1] + k[0][3]) / 2) / (0.6 * med_h)), k[0][0]))

    def adjacent(a, b) -> bool:
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a, b
        same_row = abs((ay0 + ay1) / 2 - (by0 + by1) / 2) < 0.6 * med_h
        if same_row:
            return 0 <= bx0 - ax1 < 2.0 * med_h or bx0 >= ax0 and bx0 - ax1 < 2.0 * med_h
        next_row = 0 <= by0 - ay1 < 1.2 * med_h
        overlap = min(ax1, bx1) - max(ax0, bx0)
        return next_row and overlap > 0

    out = []
    for i in range(len(toks)):
        text, conf, last = toks[i][1], toks[i][2], toks[i][0]
        out.append((text, conf))
        for j in range(i + 1, min(len(toks), i + max_tokens)):
            if not adjacent(last, toks[j][0]):
                break
            text += toks[j][1]
            conf = min(conf, toks[j][2])
            last = toks[j][0]
            out.append((text, conf))
    return out


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
        for text, conf in merge_plate_tokens(results):
            if conf < min_confidence:
                continue
            indian = normalize_indian_plate(text)
            if indian:
                # A full Indian-format match beats any partial or generic read
                candidates.append((indian, conf, 1, len(indian)))
            elif is_valid_plate_candidate(text):
                candidates.append((text, conf, 0, len(text)))

        if not candidates:
            return None

        # Prefer format matches, then longer reads (whole plate over a fragment), then confidence
        candidates.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)
        return candidates[0][0], candidates[0][1]

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
