# 🤖 Vision Model for Document Extraction API

This repository contains a robust, layout-aware document processing and field extraction API built using **Flask, Google Cloud Vision OCR, LayoutParser, and the Gemini 2.5 Flash model** for structured data extraction.

It is specifically optimized for extracting key information from **engineering forms, invoices, and checklists**, leveraging hybrid techniques:

**Vision OCR + layout segmentation + deterministic regex pre-extraction + LLM-based reasoning**

---

## ✨ Key Features

### 🧭 Layout-Aware OCR
Uses **LayoutParser** and **PaddleOCR** to segment documents (like PDFs or images) into regions:
- Text  
- Title  
- Table  
- List  

OCR is then applied region-wise for better accuracy.

---

### 🖼️ Enhanced Image Preprocessing
Applies:
- Adaptive Histogram Equalization (**CLAHE**)  
- Noise reduction  
- Adaptive sharpening  

This significantly improves OCR accuracy, especially for **low-contrast and handwritten text**.

---

### ☁️ Google Cloud Vision Integration
Uses the **Vision API** for highly accurate OCR, with **adaptive switching** between:
- `text_detection`
- `document_text_detection`

Switching is based on detected **region size and layout type**.

---

### 🧠 Intelligent Field Extraction (Gemini 2.5 Flash)

#### ✅ Deterministic Prefill
Pre-extracts highly predictable fields using regex:
- RFI No
- Date
- Structure ID
- Span ID

#### ✅ Contextual Reasoning
Uses:
- Detailed prompts
- Region summaries
- Layout cues  
to accurately infer **field-value mappings** from cleaned OCR text.

---

### 🔌 API Interface
A clean and simple **Flask-based REST API** for:
- Document upload
- OCR processing
- Structured field extraction

---

### 🌍 Ngrok Support
Optional setup for exposing the local Flask API using a **public URL** via Ngrok.

---

## 🛠️ Setup and Installation

### ✅ Prerequisites

- Python **3.8+**
- **Google Cloud Platform Project**  
  (Required for Vision API and Service Account Key JSON)
- **Gemini API Key**  
  (Obtain from **Google AI Studio**)

---

## 📦 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Vision-Model-for-Document-Extraction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
⚙️ 2. Configuration

Set your environment variables:

Variable	Description
GEMINI_API_KEY	Your Google Gemini API Key
GEMINI_MODEL_NAME	(Optional) Defaults to models/gemini-2.5-flash
LAYOUTPARSER_AVAILABLE	Automatically detected
⚠️ Note on Google Cloud Vision Key

The Google Cloud Vision Service Account Key is configured via an API endpoint after the server starts:

POST /configure_key

🚀 Running the API

The application runs on Flask and exposes its endpoints on port 5000.

python main.py


Upon startup, the console will show:

Local URL:  http://127.0.0.1:5000  
Public URL: (If pyngrok is installed, an Ngrok public URL will be printed)

🌐 API Endpoints

The core workflow follows three main steps:

Configure Key → Upload Document → Extract Fields

✅ 1. Configure Google Cloud Vision Key

Upload your Service Account Key JSON file to initialize the OCR engine.

Endpoint: POST /configure_key

Content Type: multipart/form-data

Field: key_file (JSON key file)

✅ 2. Upload Document and Perform OCR

Uploads a PDF or image file and runs the layout-aware OCR process.

Endpoint: POST /upload_invoice

Content Type: multipart/form-data

Field: invoice_file (PDF/Image, max 16MB)

✅ Success Response

{
  "status": "success",
  "message": "Invoice processed successfully",
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "pages_processed": 1,
  "regions_detected": 15
}

✅ 3. Extract Structured Fields

Uses the session_id and a list of desired field names to extract structured data using Gemini.

Endpoint: POST /extract_fields

Content Type: application/json

📤 Request Body

{
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "fields": [
    "Project Name",
    "RFI No",
    "Structure ID",
    "Date of Installation",
    "Contractor"
  ]
}


✅ Success Response

{
  "Project Name": "Highway Bridge 34",
  "RFI No": "0000220949",
  "Structure ID": "CH211",
  "Date of Installation": "10/05/2025",
  "Contractor": "ABC Engineering"
}

✅ Use Cases

Engineering Form Digitization

Invoice Processing

Construction Checklists

Field Inspection Reports

Semi-Structured PDF/Image Data Automation

📌 Tech Stack Summary

Backend: Flask

OCR: Google Cloud Vision, PaddleOCR

Layout: LayoutParser

LLM: Gemini 2.5 Flash

Tunneling: Ngrok (Optional)
