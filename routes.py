# ======================================
# routes.py — Final, Enhanced Version (Invoice OCR + Field Extractor)
# ======================================

import os
import tempfile
import json
from flask import Flask, request, jsonify
from pdf2image import convert_from_path
from datetime import datetime

# ===== Local Imports =====
from config import MAX_CONTENT_LENGTH, LAYOUTPARSER_AVAILABLE
from engines import EnhancedOCREngine
from services import FieldQuestionExtractionService
from utils import (
    ocr_engine,
    service_account_key_path,
    global_lock,
    setup_ngrok,
    cleanup_ngrok,
    extracted_data_store,
    create_session,
)

# Initialize Flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ngrok setup (optional)
try:
    NGROK_AVAILABLE = setup_ngrok() is not None
except Exception:
    NGROK_AVAILABLE = False


# ======================================
# BASIC ROUTES
# ======================================

@app.route("/")
def home():
    return jsonify({
        "message": "Invoice OCR & Field Extraction API (Gemini + Cloud Vision + LayoutParser)",
        "features": {
            "layout_parser": LAYOUTPARSER_AVAILABLE,
            "ngrok": NGROK_AVAILABLE
        },
        "endpoints": {
            "configure_key": "POST /configure_key",
            "upload_invoice": "POST /upload_invoice",
            "extract_fields": "POST /extract_fields",
            "health": "GET /health"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    with global_lock:
        status = "healthy" if ocr_engine else "key_not_configured"
    return jsonify({
        "status": status,
        "layout_parser_available": LAYOUTPARSER_AVAILABLE
    })


# ======================================
# CONFIGURE GOOGLE KEY
# ======================================

@app.route("/configure_key", methods=["POST"])
def configure_key():
    global ocr_engine, service_account_key_path

    if "key_file" not in request.files:
        return jsonify({"error": "No key file provided"}), 400

    key_file = request.files["key_file"]
    if key_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            key_file.save(tmp.name)
            key_path = tmp.name

        # Initialize Vision OCR Engine
        test_engine = EnhancedOCREngine(key_path)
        with global_lock:
            service_account_key_path = key_path
            ocr_engine = test_engine

        return jsonify({
            "status": "success",
            "message": "Service account key configured successfully",
            "layout_parser": LAYOUTPARSER_AVAILABLE
        })
    except Exception as e:
        return jsonify({"error": f"Key configuration failed: {str(e)}"}), 500


# ======================================
# UPLOAD INVOICE + OCR EXTRACTION
# ======================================

@app.route("/upload_invoice", methods=["POST"])
def upload_invoice():
    global ocr_engine, extracted_data_store

    with global_lock:
        if not ocr_engine:
            return jsonify({"error": "Service account key not configured"}), 400

    if "invoice_file" not in request.files:
        return jsonify({"error": "No invoice file provided"}), 400

    invoice_file = request.files["invoice_file"]
    if invoice_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            invoice_file.save(tmp.name)
            pdf_path = tmp.name

        pages = convert_from_path(pdf_path, dpi=300)
        os.unlink(pdf_path)

        print(f"📄 Processing {len(pages)} pages using Google Vision + LayoutParser...")

        all_text_blocks = []
        regions_meta = []
        total_blocks = 0

        for i, page in enumerate(pages):
            result = ocr_engine.extract_with_layout_vision(page)

            if result.get("cleaned_text"):
                all_text_blocks.append(result["cleaned_text"])
                total_blocks += result.get("blocks_processed", 0)

                # Add layout summary for region analysis
                regions_meta.append({
                    "page": i + 1,
                    "blocks_processed": result.get("blocks_processed", 0),
                    "chars": len(result["cleaned_text"]),
                })

                print(f"✅ Page {i+1}: {result.get('blocks_processed', 0)} regions, {len(result['cleaned_text'])} chars")
            else:
                print(f"⚠️ Page {i+1}: no text extracted")

        combined_text = "\n\n".join(all_text_blocks).strip()

        if not combined_text:
            return jsonify({"error": "No text could be extracted from the document"}), 400

        # Create new session with combined results
        session_id = create_session(
            extracted_text=combined_text,
            pages_processed=len(pages),
            total_blocks=total_blocks,
            regions_meta=regions_meta
        )

        print(f"📊 Extraction Summary → Session: {session_id} | Pages: {len(pages)} | Blocks: {total_blocks}")
        print(f"📈 Total text length: {len(combined_text)} characters")

        return jsonify({
            "status": "success",
            "message": "Invoice processed successfully",
            "session_id": session_id,
            "pages_processed": len(pages),
            "regions_detected": total_blocks
        })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# ======================================
# GEMINI FIELD EXTRACTION
# ======================================

extraction_service = FieldQuestionExtractionService()


@app.route("/extract_fields", methods=["POST"])
def extract_fields():
    """Extract specified fields (keywords) from an uploaded invoice."""
    try:
        data = request.get_json()
        fields = data.get("fields")
        session_id = data.get("session_id")

        if not fields:
            return jsonify({"error": "No fields provided"}), 400
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400

        session_data = extracted_data_store.get(session_id)
        if not session_data or "cleaned_text" not in session_data:
            return jsonify({"error": "No uploaded invoice found for this session"}), 404

        # ✅ FIXED: assign cleaned_text first before printing
        cleaned_text = session_data["cleaned_text"]

        print("\n===== CLEANED OCR TEXT BEFORE GEMINI =====")
        print(cleaned_text[:5000])  # print first 5000 chars to avoid flooding logs
        print("==========================================\n")

        # Run Gemini field extraction
        result = extraction_service.extract_fields(cleaned_text, fields)

        return jsonify(result)

    except Exception as e:
        print(f"❌ Field extraction failed: {e}")
        return jsonify({"error": f"Field extraction failed: {str(e)}"}), 500



# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    app.run(debug=True)
