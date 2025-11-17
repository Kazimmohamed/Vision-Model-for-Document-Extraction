# ================================
# services.py — Intelligent Field Extractor
# (Layout-Aware + Context-Aware + Deterministic Prefill)
# FINAL PRODUCTION VERSION
# ================================

import json
import re
import os
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class FieldQuestionExtractionService:
    """
    Field extractor using Google Gemini for structured field-value mapping.
    Integrates:
    - Deterministic regex-based pre-extraction
    - Context-aware understanding of engineering checklists
    - Layout-aware region summaries
    - Flat JSON output
    """

    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.DEBUG_OCR_PRINT = os.environ.get("DEBUG_OCR_PRINT", "false").lower() in ("1", "true", "yes")

    # ==============================================================
    # 🧠 Main Function: Layout- and Context-Aware Field Extraction
    # ==============================================================
    def extract_fields(self, cleaned_text: str, fields: list, regions_meta=None):
        """
        Extract specified fields from OCR text using hybrid (regex + LLM) reasoning.
        Returns flat JSON: {"Field": "Value"}
        """
        if not cleaned_text or not cleaned_text.strip():
            return {"error": "Empty or invalid document text"}
        if not fields:
            return {"error": "No fields provided"}

        # 1️⃣ Pre-clean & Prefill deterministic values
        text = self._pre_clean_text(cleaned_text)
        prefill = self._regex_preextract(text)
        if self.DEBUG_OCR_PRINT:
            print("\n=== PREFILL (regex) ===")
            print(prefill)

        # 2️⃣ Optional: summarize top few regions
        region_summary = ""
        if regions_meta and isinstance(regions_meta, list):
            lines = []
            for r in regions_meta[:5]:
                rtext = (r.get("region_text") or "")[:200].replace("\n", "\\n")
                lines.append(f"[{r.get('index','?')}] {r.get('type','Text')}: {rtext}")
            region_summary = "\n".join(lines)

        # 3️⃣ Inject context header (to simulate human-level understanding)
        context_header = """
You are an intelligent field extraction system.
You are reading OCR text that represents an engineering form or checklist.
Treat the text as a structured document — not plain text.
Infer logical groupings even when alignment or OCR spacing is broken.

Typical document content includes:
• Project metadata (project name, client, contractor, location, chainage)
• Component details (structure ID, span ID, bearing IDs)
• Measurement results and tolerances
• Engineer sign-offs and dates

Think like an engineer reading the checklist.
Associate field labels and nearby values even if on separate lines.
Never hallucinate; leave blank if uncertain.
Preserve exact formatting (including leading zeros).
Output valid JSON only.
"""

        # 4️⃣ Prompt construction
        prefill_note = ""
        if prefill:
            prefill_note = "Already found (regex prefill): " + json.dumps(prefill) + "\n"

        rules = (
            "Rules:\n"
            "1) Prefer deterministic regex results when clearly valid.\n"
            "2) CH### → Structure ID, P##-P## → Span ID.\n"
            "3) Never invent data; leave field empty if unsure.\n"
            "4) Prefer values from same region as label.\n"
            "5) Preserve number formats and units.\n"
        )

        one_shot_example = (
            "Example:\n"
            "Input:\n"
            "[1] Text: RFI No: 0000220949\\n[2] Text: CH211 P17-P18\n"
            "Output JSON: {\"RFI No\":\"0000220949\",\"Structure ID\":\"CH211\",\"Span ID\":\"P17-P18\"}\n"
        )

        fields_str = "\n".join([f"- {f}" for f in fields])

        prompt = (
            f"{context_header}\n\n"
            f"{prefill_note}"
            f"OCR Text (truncated):\n{text[:6000]}\n\n"
            f"Region summary (top 5 regions):\n{region_summary}\n\n"
            f"{rules}\n"
            f"{one_shot_example}\n"
            f"Fields to extract:\n{fields_str}\n\n"
            "Output only valid JSON mapping each field to its value."
        )

        if self.DEBUG_OCR_PRINT:
            print("\n=== PROMPT (first 4000 chars) ===")
            print(prompt[:4000])

        # 5️⃣ Call Gemini API
        try:
            response = self.model.generate_content(prompt)
            if not response or not getattr(response, "text", None):
                return {"error": "Empty response from Gemini"}

            text_out = response.text.strip()
            if self.DEBUG_OCR_PRINT:
                print("\n=== GEMINI RAW OUTPUT ===")
                print(text_out)

            json_match = re.search(r'\{[\s\S]*\}', text_out)
            if not json_match:
                text_fixed = text_out.replace("```json", "").replace("```", "").strip()
                try:
                    model_data = json.loads(text_fixed)
                except Exception:
                    return {"error": "No valid JSON found", "raw_output": text_out}
            else:
                model_data = json.loads(self._clean_json(json_match.group(0)))

        except Exception as e:
            return {"error": f"Gemini processing failed: {str(e)}"}

        # 6️⃣ Merge Prefill and Model Output
        final = {}
        for field in fields:
            val = ""
            if isinstance(model_data, dict) and model_data.get(field):
                val = str(model_data.get(field)).strip()
            elif field in prefill:
                val = prefill[field]
            final[field] = val

        return final  # ✅ flat JSON

    # ==============================================================
    # 🔎 Deterministic Regex Extractor
    # ==============================================================
    def _regex_preextract(self, text: str) -> dict:
        """Extract predictable fields deterministically (date, RFI, CH#, Span)."""
        out = {}
        if not text:
            return out

        # Date (supports multiple separators)
        m = re.search(r'\b(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})\b', text)
        if m:
            out["Date of Installation"] = m.group(1).strip()

        # RFI No
        m = re.search(r'RFI\s*No[:\s-]*([A-Za-z0-9\-/]+)', text, re.I)
        if m:
            out["RFI No"] = m.group(1).strip()

        # Structure ID (CH###)
        m = re.search(r'\bCH\s*\d{1,5}\b', text, re.I)
        if m:
            out["Structure ID"] = m.group(0).replace(" ", "").strip()

        # Span ID (P##-P##)
        m = re.search(r'\bP\d{1,3}-P\d{1,3}\b', text, re.I)
        if m:
            out["Span ID"] = m.group(0).strip()

        # Fallback numeric RFI
        m = re.search(r'\b0{3,}\d{4,10}\b', text)
        if m and "RFI No" not in out:
            out["RFI No"] = m.group(0).strip()

        return out

    # ==============================================================
    # 🧹 Text Cleaning Helpers
    # ==============================================================
    def _pre_clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving line structure."""
        t = text
        t = re.sub(r'\r+', '', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        t = re.sub(r'\s{3,}', '  ', t)
        return t.strip()

    def _clean_json(self, json_str: str) -> str:
        """Fixes Gemini JSON formatting quirks."""
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*\]", "]", json_str)
        json_str = json_str.replace("“", '"').replace("”", '"')
        json_str = json_str.replace("‘", "'").replace("’", "'")
        json_str = json_str.replace("\u00a0", " ")
        return json_str.strip()
