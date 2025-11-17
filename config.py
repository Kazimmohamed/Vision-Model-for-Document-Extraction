#  config.py
import os

# ===============================
# Flask Configuration
# ===============================
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit


# ===============================
# Google Gemini Configuration
# ===============================
# ✅ Prefer environment variable for security; fallback to default only for local testing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBy-LVsVLviMcLkd9gpZDqGSabv7Dva7NM")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")



LAYOUTPARSER_AVAILABLE = False
NGROK_AVAILABLE = False

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    pass

try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    pass


# ===============================
# Debug Summary (optional)
# ===============================
if __name__ == "__main__":
    print(f"Gemini Model: {MODEL_NAME}")
    print(f"LayoutParser Available: {LAYOUTPARSER_AVAILABLE}")
    print(f"Ngrok Available: {NGROK_AVAILABLE}")
