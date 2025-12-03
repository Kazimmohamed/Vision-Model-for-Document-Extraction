# 🤖 Vision Model for Document Extraction API

This repository contains a robust, **layout-aware document processing and field extraction API** built using:

- **Flask**
- **Google Cloud Vision OCR**
- **LayoutParser**
- **Gemini 2.5 Flash Model**

It is specifically optimized for extracting key information from **engineering forms, invoices, and checklists**, using a **hybrid pipeline**:

> Vision OCR + Layout Segmentation + Deterministic Regex Pre-Extraction + LLM-Based Reasoning

---

## ✨ Key Features

### 🧭 Layout-Aware OCR
- Uses **LayoutParser** and **PaddleOCR**
- Segments PDFs or images into:
  - Text
  - Title
  - Table
  - List  
- OCR is applied **region-wise** for higher accuracy

### 🖼️ Enhanced Image Preprocessing
Improves OCR results using:
- Adaptive Histogram Equalization (**CLAHE**)
- Noise Reduction
- Adaptive Sharpening  
Especially effective for **low-contrast and handwritten text**

### ☁️ Google Cloud Vision Integration
- Uses the **Vision API** for highly accurate OCR
- Automatically switches between:
  - `text_detection`
  - `document_text_detection`  
based on **region size**

### 🧠 Intelligent Field Extraction (Gemini 2.5 Flash)
- **Deterministic Prefill**  
  Regex-based extraction for predictable fields:
  - RFI No
  - Date
  - Structure ID
  - Span ID
- **Contextual Reasoning**  
  Uses:
  - Region summaries
  - Layout cues
  - Cleaned OCR text  
  to infer accurate **field-value mappings**

### 🔌 API Interface
- Simple **Flask-based REST API**
- Supports:
  - File upload
  - OCR processing
  - Structured field extraction

### 🌍 Ngrok Support
- Optional public exposure of the API using **ngrok**

---

## 🛠️ Setup and Installation

### ✅ Prerequisites

- Python **3.8+**
- Google Cloud Platform Project  
  (For Vision API + Service Account Key JSON)
- Gemini API Key  
  (Get it from **Google AI Studio**)

---

## 📦 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Vision-Model-for-Document-Extraction
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

