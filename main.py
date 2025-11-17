from routes import app

from utils import setup_ngrok, cleanup_ngrok
from config import LAYOUTPARSER_AVAILABLE, NGROK_AVAILABLE
import atexit


if __name__ == '__main__':
    atexit.register(cleanup_ngrok)
    print("🚀 Starting Enhanced Invoice OCR & Chat Flask API...")
    print(f"📊 LayoutParser Available: {LAYOUTPARSER_AVAILABLE}")
    
    public_url = setup_ngrok(5000)
    print("🌐 Running locally at: http://127.0.0.1:5000")
    if public_url:
        print(f"🌍 Public URL: {public_url}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)