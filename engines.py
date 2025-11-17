# engines.py
# FINALIZED — EnhancedOCREngine with layout-aware segmentation,
# adaptive preprocessing, and region-size-based Vision mode switching.
# CHANGELOG (applied):
# - Store raw region_text (preserve newlines) in regions_meta, capped at 3000 chars.
# - Do not strip/collapse newlines when appending region_text to all_text_blocks.
# - Safe length handling for region_text usage to avoid None issues.
# - Kept combined text building as "\n\n".join(all_text_blocks).
# NOTE: Keep preprocessors.postprocessor from removing newlines (separate file edit).

import os
import io
import cv2
import numpy as np
from PIL import Image
from google.cloud import vision_v1
from typing import List, Dict, Any

# ============================
# Local Imports
# ============================
from config import LAYOUTPARSER_AVAILABLE
from preprocessors import DocumentPreprocessor, EnhancedTextPostProcessor

# ============================
# Helper functions
# ============================
def _bbox_to_tuple(block) -> tuple:
    """Convert layout block coordinates to integer bbox (x1,y1,x2,y2)."""
    try:
        return (
            int(block.x_1),
            int(block.y_1),
            int(block.x_2),
            int(block.y_2),
        )
    except Exception:
        return (0, 0, 0, 0)


def _format_region_marker(rtype: str, idx: int, bbox: tuple) -> str:
    """Create a small region marker string to inject into concatenated text."""
    x1, y1, x2, y2 = bbox
    return f"[REGION:{rtype}|{idx}|bbox:{x1},{y1},{x2},{y2}]"


def _adaptive_hist_equalize(pil_img: Image.Image, std_threshold: float = 30.0) -> Image.Image:
    """If image brightness/contrast is low (likely faded handwriting), apply histogram equalization."""
    try:
        gray = pil_img.convert("L")
        np_img = np.array(gray)
        if np.std(np_img) < std_threshold:
            eq = cv2.equalizeHist(np_img)
            boosted = cv2.convertScaleAbs(eq, alpha=1.5, beta=8)
            return Image.fromarray(boosted)
        return pil_img
    except Exception as e:
        print(f"Adaptive enhancement failed: {e}")
        return pil_img


# ============================
# Layout Parser Engine
# ============================
class LayoutParserEngine:
    """Layout detection using LayoutParser + PaddleOCR."""
    def __init__(self):
        if not LAYOUTPARSER_AVAILABLE:
            raise ImportError("LayoutParser not available. Install with: pip install layoutparser[ocr] paddleocr")

        import layoutparser as lp

        try:
            self.model = lp.PaddleDetectionLayoutModel(
                config_path='lp://PubLayNet/PPOCRv3/config',
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                enforce_cpu=True
            )
            print("✅ LayoutParser (PaddleOCR) initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR layout model: {e}")

    def detect_regions(self, image: Image.Image) -> List[Any]:
        """Detect text regions in the document and return a list of layout regions."""
        try:
            image_np = np.array(image.convert("RGB"))
            layouts = self.model.detect(image_np)

            text_regions = [layout for layout in layouts if layout.type in ["Text", "Title", "List", "Table"]]
            text_regions.sort(key=lambda l: (getattr(l.block, "y_1", 0), getattr(l.block, "x_1", 0)))
            return text_regions

        except Exception as e:
            print(f"Layout detection error: {e}")
            return []

    def crop_regions(self, image: Image.Image, regions: List[Any]) -> List[Image.Image]:
        """Crop detected text regions from the image."""
        cropped_images = []
        image_np = np.array(image.convert("RGB"))

        for i, region in enumerate(regions):
            try:
                x1, y1, x2, y2 = _bbox_to_tuple(region.block)
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.05)
                x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                x2, y2 = min(image_np.shape[1], x2 + padding_x), min(image_np.shape[0], y2 + padding_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped = image_np[y1:y2, x1:x2]
                cropped_images.append(Image.fromarray(cropped))
            except Exception as e:
                print(f"Region {i} crop failed: {e}")

        return cropped_images


# ============================
# Enhanced OCR Engine
# ============================
class EnhancedOCREngine:
    def __init__(self, key_path: str = None):
        if key_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        self.vision_client = vision_v1.ImageAnnotatorClient()
        self.preprocessor = DocumentPreprocessor()
        self.postprocessor = EnhancedTextPostProcessor()

        self.layout_engine = None
        if LAYOUTPARSER_AVAILABLE:
            try:
                self.layout_engine = LayoutParserEngine()
            except Exception as e:
                print(f"⚠️ LayoutParser initialization failed: {e}")
        else:
            print("⚠️ LayoutParser not available, using full-page OCR fallback")

    # ---------------------------------------------
    # Main OCR Function
    # ---------------------------------------------
    def extract_with_layout_vision(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using LayoutParser + Google Vision."""
        all_text_blocks = []
        regions_meta = []

        try:
            if self.layout_engine:
                regions = self.layout_engine.detect_regions(image)
                if regions:
                    cropped_regions = self.layout_engine.crop_regions(image, regions)
                    print(f"📄 Detected {len(cropped_regions)} text regions")

                    for i, (region_obj, region_image) in enumerate(zip(regions, cropped_regions)):
                        enhanced_image = _adaptive_hist_equalize(region_image)
                        processed_image = self.preprocessor.preprocess(enhanced_image)

                        # Note: PIL size is (width, height)
                        w, h = processed_image.size[0], processed_image.size[1]
                        prefer_text = (h < 300 or w < 600)
                        region_text = self._vision_ocr_from_pil(processed_image, prefer_text_detection=prefer_text) or ""

                        bbox = _bbox_to_tuple(region_obj.block)
                        region_marker = _format_region_marker(region_obj.type, i + 1, bbox)

                        # Preserve newlines when appending to all_text_blocks (do NOT strip)
                        if region_text:
                            all_text_blocks.append(f"{region_marker}\n{region_text}")

                        # Store raw region_text in regions_meta, preserve newlines, cap to 3000 chars
                        safe_region_text = (region_text or "")[:3000]
                        regions_meta.append({
                            "index": i + 1,
                            "type": getattr(region_obj, "type", "Text"),
                            "bbox": bbox,
                            "chars": len(region_text) if region_text else 0,
                            "region_text": safe_region_text  # preserve newlines
                        })

                        print(f"  Region {i+1} ({getattr(region_obj,'type','Text')}): {len(region_text)} chars (mode={'text' if prefer_text else 'doc'})")

                else:
                    print("⚠️ No layout regions detected, using full-page OCR")
                    processed_image = self.preprocessor.preprocess(image)
                    page_text = self._vision_ocr_from_pil(processed_image) or ""
                    if page_text:
                        all_text_blocks.append(f"[REGION:FULL_PAGE|1|bbox:0,0,0,0]\n{page_text}")
                        regions_meta.append({
                            "index": 1,
                            "type": "FullPage",
                            "bbox": (0, 0, 0, 0),
                            "chars": len(page_text),
                            "region_text": (page_text or "")[:3000]
                        })
            else:
                processed_image = self.preprocessor.preprocess(image)
                page_text = self._vision_ocr_from_pil(processed_image) or ""
                if page_text:
                    all_text_blocks.append(f"[REGION:FULL_PAGE|1|bbox:0,0,0,0]\n{page_text}")
                    regions_meta.append({
                        "index": 1,
                        "type": "FullPage",
                        "bbox": (0, 0, 0, 0),
                        "chars": len(page_text),
                        "region_text": (page_text or "")[:3000]
                    })

            if all_text_blocks:
                # Important: keep double-newline join to preserve region separation
                combined_text = "\n\n".join(all_text_blocks)
                final_cleaned = self.postprocessor.clean_text(combined_text)
                return {
                    "raw_text": combined_text,
                    "cleaned_text": final_cleaned,
                    "blocks_processed": len(all_text_blocks),
                    "regions_meta": regions_meta
                }

            return {"raw_text": "", "cleaned_text": "", "blocks_processed": 0, "regions_meta": []}

        except Exception as e:
            print(f"OCR extraction error: {e}")
            return {"raw_text": "", "cleaned_text": "", "blocks_processed": 0, "regions_meta": [], "error": str(e)}

    # ---------------------------------------------
    # Vision API Helper
    # ---------------------------------------------
    def _vision_ocr_from_pil(self, pil_img: Image.Image, prefer_text_detection: bool = False) -> str:
        """Run Vision OCR with automatic fallback."""
        try:
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="JPEG", quality=95)
            image_content = img_bytes.getvalue()
            vision_image = vision_v1.Image(content=image_content)

            try:
                if prefer_text_detection:
                    response = self.vision_client.text_detection(image=vision_image)
                else:
                    response = self.vision_client.document_text_detection(image=vision_image)
            except Exception as api_e:
                print(f"Vision API primary failed ({api_e}), trying fallback.")
                response = self.vision_client.document_text_detection(image=vision_image)

            if getattr(response, "error", None) and response.error.message:
                print(f"Vision API Error: {response.error.message}")
                return ""

            raw_text = ""
            if getattr(response, "full_text_annotation", None) and response.full_text_annotation.text:
                raw_text = response.full_text_annotation.text
            elif getattr(response, "text_annotations", None) and len(response.text_annotations) > 0:
                raw_text = response.text_annotations[0].description or ""

            return raw_text

        except Exception as e:
            print(f"Single region OCR error: {e}")
            return ""
