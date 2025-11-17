# preprocessors.py — Adaptive Enhancement + Intelligent Text Post-Processing
# FINALIZED VERSION — Drop-in replacement.
# CHANGELOG (applied):
# - Removed aggressive digit-merging (re.sub(r'(\d)\s+(\d)', ...)).
# - Strengthened whitespace cleanup to preserve layout newlines.
# - Ensured [REGION:...] markers remain exactly as in OCR.
# - Preserved region-level newlines to support layout-aware LLM prompts.

import re
import cv2
import numpy as np
from PIL import Image


# ================================
# Document Preprocessor
# ================================
class DocumentPreprocessor:
    """
    Handles document-level image preprocessing for OCR.
    Includes adaptive enhancement for handwritten / low-contrast areas.
    """

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess a PIL image for OCR:
        - Converts to grayscale
        - Adaptive histogram equalization (CLAHE) or global equalization for faint regions
        - Optional local contrast/brightness boost for handwriting
        - Noise reduction
        - Adaptive sharpening (based on contrast)
        """
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            img_np = np.array(image)

            # Compute contrast estimate
            std_dev = float(np.std(img_np))

            # For very low contrast (likely faint handwriting), use global equalization + boost
            if std_dev < 25:
                eq = cv2.equalizeHist(img_np)
                boosted = cv2.convertScaleAbs(eq, alpha=1.8, beta=10)
                enhanced = boosted
            else:
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
                enhanced = clahe.apply(img_np)

            # Light denoise for scanned grain
            denoised = cv2.medianBlur(enhanced, 3)

            # Adaptive sharpening intensity based on texture variance
            sigma = 1.5 if std_dev > 40 else 1.0
            gaussian = cv2.GaussianBlur(denoised, (0, 0), sigma)
            sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            return Image.fromarray(sharpened)

        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return image


# ================================
# Text Postprocessor
# ================================
class EnhancedTextPostProcessor:
    """
    Cleans and normalizes OCR text output for consistent downstream processing.
    Handles:
    - Common OCR character misreads (O→0, I→1, etc.)
    - Date normalization
    - Region marker preservation
    - Whitespace cleanup (while preserving layout)
    """

    def __init__(self):
        self.ocr_corrections = {
            r'\bO(?=\d)': '0',            # O1 -> 01
            r'\bI(?=\d)': '1',            # I1 -> 11
            r'\b0I\b': '01',
            r'\bP16017\b': 'P16-P17',
            r'\bP(\d{2})0(\d{2})\b': r'P\1-\2',
            r'\\[lI]\b': '1',
            r'[\[\]oO]{3,}': '0',
        }

        # Keep punctuation needed for layout and engineering notation
        self.keep_characters = r'\.\-\/\@\:\,\%\(\)\#\&\[\]\|'

    # ------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize OCR text safely:
        - Preserves [REGION:...] markers
        - Applies OCR corrections conservatively
        - Removes unwanted symbols without destroying layout structure
        - Normalizes whitespace but keeps region separators (\n\n)
        """
        if not text:
            return ""

        # 1) Protect [REGION:...] markers
        region_markers = re.findall(r'\[REGION:[^\]]+\]', text)
        placeholder_map = {f"__REGION_{i}__": rm for i, rm in enumerate(region_markers)}
        for ph, mk in placeholder_map.items():
            text = text.replace(mk, ph)

        # 2) Apply OCR fixups
        for pattern, repl in self.ocr_corrections.items():
            try:
                text = re.sub(pattern, repl, text)
            except re.error:
                continue

        # 3) Whitespace normalization (preserve layout newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # limit long newline bursts
        lines = text.split('\n')
        # Remove excessive inner spaces but do not flatten lines
        lines = [re.sub(r'\s+', ' ', ln).strip() for ln in lines]
        text = '\n'.join(lines)

        # 4) Remove illegal/control chars while keeping layout punctuation
        try:
            text = re.sub(fr'[^\w\s{self.keep_characters}]', ' ', text)
        except re.error:
            text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)

        # 5) Restore [REGION:...] markers
        for ph, mk in placeholder_map.items():
            text = text.replace(ph, mk)

        # 6) Normalize date formats
        text = re.sub(r'(\b\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{2,4}\b)', r'\1/\2/\3', text)

        # ✅ 7) DO NOT merge digits (removed unsafe rule)
        # Example of removed rule:
        # text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

        return text.strip()

    # ------------------------------------------------------------
    def concatenate_blocks(self, block_texts):
        """
        Concatenate multiple OCR blocks while preserving region boundaries.
        Adds double-newline between blocks to give LLMs layout cues.
        """
        full_text = ""
        for block in block_texts:
            if not block or not block.strip():
                continue
            if full_text and not full_text.endswith("\n\n"):
                full_text += "\n\n"
            full_text += block.strip()

        return full_text.strip()
