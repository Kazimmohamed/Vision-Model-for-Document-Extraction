# Save as verify_env.py
import layoutparser as lp
from paddleocr import PaddleOCR

print("⏳ Checking PaddleOCR + LayoutParser environment...")

try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("✅ PaddleOCR initialized successfully.")
except Exception as e:
    print("❌ PaddleOCR failed:", e)

try:
    model = lp.PaddleDetectionLayoutModel(
        "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config", enforce_cpu=True
    )
    print("✅ LayoutParser model loaded successfully.")
except Exception as e:
    print("❌ LayoutParser failed:", e)
