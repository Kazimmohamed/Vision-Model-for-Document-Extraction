# ================================
# utils.py — Global Session Store & Utilities
# ================================

import uuid
from datetime import datetime
from threading import Lock
import json
import atexit
from config import NGROK_AVAILABLE

# ================================
# GLOBALS & SESSION STORAGE
# ================================
ocr_engine = None
service_account_key_path = None

# Active sessions: per uploaded invoice/document
# Structure: { session_id: { extracted_text, created_at, total_blocks, pages_processed, regions_meta } }
sessions = {}

# Cleaned text store for Gemini structuring API
# Structure: { session_id: { cleaned_text, timestamp } }
extracted_data_store = {}

# Thread lock for concurrency safety
global_lock = Lock()


# ================================
# NGROK SUPPORT (Optional)
# ================================
ngrok_tunnel = None


def setup_ngrok(port=5000):
    """Setup ngrok tunnel for public Flask access"""
    global ngrok_tunnel
    if not NGROK_AVAILABLE:
        print("❌ Ngrok not available. Run: pip install pyngrok")
        return None
    try:
        from pyngrok import ngrok
        ngrok_tunnel = ngrok.connect(port)
        print(f"✅ Ngrok tunnel active: {ngrok_tunnel.public_url}")
        return ngrok_tunnel.public_url
    except Exception as e:
        print(f"Ngrok setup failed: {e}")
        return None


def cleanup_ngrok():
    """Disconnect ngrok tunnel safely"""
    global ngrok_tunnel
    if ngrok_tunnel and NGROK_AVAILABLE:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(ngrok_tunnel.public_url)
            print("🔒 Ngrok tunnel closed")
        except Exception as e:
            print(f"Error closing ngrok: {e}")


# Ensure ngrok closes gracefully on shutdown
atexit.register(cleanup_ngrok)


# ================================
# SESSION UTILITIES
# ================================
def create_session(extracted_text: str, pages_processed: int, total_blocks: int, regions_meta=None):
    """Creates a new session and stores extracted text & metadata."""
    global sessions, extracted_data_store
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    if regions_meta is None:
        regions_meta = []

    with global_lock:
        sessions[session_id] = {
            "extracted_text": extracted_text,
            "created_at": created_at,
            "pages_processed": pages_processed,
            "total_blocks": total_blocks,
            "regions_meta": regions_meta,
        }
        extracted_data_store[session_id] = {
            "cleaned_text": extracted_text,
            "timestamp": created_at,
        }

    print(f"✅ Session created: {session_id} | Pages: {pages_processed} | Blocks: {total_blocks}")
    return session_id


def summarize_session(session_id: str):
    """Prints a quick summary of a given session (for debugging)."""
    with global_lock:
        session = sessions.get(session_id)
        if not session:
            print(f"⚠️ Session {session_id} not found.")
            return

        print("\n📄 Session Summary:")
        print(f"  Session ID     : {session_id}")
        print(f"  Created At     : {session.get('created_at')}")
        print(f"  Pages Processed: {session.get('pages_processed')}")
        print(f"  Blocks Detected: {session.get('total_blocks')}")
        print(f"  Regions Meta   : {len(session.get('regions_meta', []))} entries")

        # Show first few regions for quick inspection
        regions_meta = session.get("regions_meta", [])
        if regions_meta:
            print("\n  🧩 Sample Regions:")
            for i, region in enumerate(regions_meta[:3]):
                bbox = region.get("bbox")
                print(f"    [{i+1}] Type={region.get('type')} | Chars={region.get('chars')} | BBox={bbox}")

        print("-" * 60)


def export_sessions_to_json(filepath="sessions_log.json"):
    """Export all active session summaries to a JSON file."""
    with global_lock:
        try:
            export_data = {
                sid: {
                    "created_at": s.get("created_at"),
                    "pages_processed": s.get("pages_processed"),
                    "total_blocks": s.get("total_blocks"),
                    "regions_meta_count": len(s.get("regions_meta", []))
                }
                for sid, s in sessions.items()
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            print(f"💾 Sessions exported successfully → {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to export sessions: {e}")
