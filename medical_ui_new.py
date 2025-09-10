# smart_medicine_app.py
"""
Smart Medicine Assistant — GPU-first Donut OCR + EasyOCR + Tesseract fallback
- Auto-loads .env (python-dotenv)
- Primary OCR: Hugging Face Donut (uses GPU if available & HF_TOKEN set)
- Fallback OCR: EasyOCR (uses GPU if torch.cuda available)
- Final fallback: pytesseract (CPU)
- Heuristic extraction always available to guarantee results
- IBM Granite present as a safe stub (non-blocking side-character)
- SerpAPI optional for assistant fallback
- Defensive: missing packages or tokens won't crash the app
- UI shows which OCR backend was used (Donut / EasyOCR / Tesseract / Manual)
Notes:
- Install optional dependencies if you want full functionality:
    pip install python-dotenv streamlit pillow pandas numpy plotly requests
    # Optional (recommended for HF + GPU):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers huggingface_hub accelerate
    # EasyOCR and Tesseract:
    pip install easyocr pytesseract
    # For pytesseract on Linux/Windows, install tesseract-ocr system package as needed.
"""

import os
import re
import json
import sqlite3
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Load .env automatically if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------
# Environment & tokens
# --------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", None)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", None)
DB_PATH = os.environ.get("DB_PATH", "patient_data.db")

# Optional Twilio (SMS)
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", None)
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", None)
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER", None)

# Donut / model names (defaults)
DONUT_MODEL_NAME = os.environ.get("DONUT_MODEL_NAME", "chinmays18/medical-prescription-ocr")
NER_MODEL_NAME = os.environ.get("NER_MODEL_NAME", "d4data/biomedical-ner-all")

# CPU/CUDA override (if you want to force CPU set USE_CUDA=0 in .env)
USE_CUDA_ENV = os.environ.get("USE_CUDA", None)
FORCE_CPU = False
if USE_CUDA_ENV is not None and str(USE_CUDA_ENV).strip().lower() in ("0", "false", "no"):
    FORCE_CPU = True

def check_hf_token() -> Tuple[bool, str]:
    if not HF_TOKEN:
        return False, ("HF token not set. Place HF_TOKEN in .env or environment. "
                       "Without it Donut OCR and HF inference will be disabled.")
    if not HF_TOKEN.startswith("hf_"):
        return False, ("HF token doesn't look valid (should start with 'hf_').")
    return True, "HF token found."

# --------------------------
# Optional heavy libs (guarded)
# --------------------------
HF_AVAILABLE = False
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
HUB_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    # Import transformers components
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from huggingface_hub import InferenceClient, login
    TRANSFORMERS_AVAILABLE = True
    HUB_AVAILABLE = True
    HF_AVAILABLE = True
except Exception:
    # We'll still try to proceed without HF
    HF_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    HUB_AVAILABLE = False
    try:
        # torch might still be importable separately
        import torch  # reattempt safe import
        TORCH_AVAILABLE = True
    except Exception:
        TORCH_AVAILABLE = False

# EasyOCR & pytesseract (optional fallback)
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# Twilio optional
TWILIO_AVAILABLE = False
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# --------------------------
# Simple internal IBM Granite stub (single module — side-character)
# --------------------------
class GraniteSideCharacter:
    """
    Granite side-character wrapper.
    - If ibm_watsonx_ai SDK present, attempt lazy initialization on demand.
    - Otherwise provide safe stub replies.
    """
    def __init__(self):
        self.client = None
        self.sdk_available = False
        try:
            import ibm_watsonx_ai  # presence check only
            self.sdk_available = True
        except Exception:
            self.sdk_available = False
        self.api_key = os.environ.get("IBM_API_KEY", None)
        self.project_id = os.environ.get("IBM_PROJECT_ID", None)
        self.region = os.environ.get("IBM_REGION", "us-south")

    def init_client(self) -> Tuple[bool, str]:
        if not self.sdk_available:
            return False, "IBM SDK not installed - using Granite stub."
        if not (self.api_key and self.project_id):
            return False, "IBM credentials (IBM_API_KEY / IBM_PROJECT_ID) not configured - using Granite stub."
        try:
            from ibm_watsonx_ai import Credentials, Model
            creds = Credentials(url=f"https://{self.region}.ml.cloud.ibm.com", api_key=self.api_key)
            model = Model(model_id="ibm/granite-guardian-3.2-5b-lora-harm-correction", credentials=creds, project_id=self.project_id)
            self.client = model
            return True, "Granite client initialized."
        except Exception as e:
            return False, f"Failed to initialize Granite: {e}"

    def correct_text(self, text: str) -> Tuple[bool, str]:
        if not self.client:
            ok, msg = self.init_client()
            if not ok:
                # tiny heuristic cleanup as stub
                cleaned = re.sub(r"[^A-Za-z0-9\s\-,.%/()#mg]", " ", (text or ""))
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
                if len(cleaned) > 0:
                    return True, f"[Stub-cleaned text] {cleaned}"
                return False, "Granite not configured; stub clean produced no output."
        try:
            prompt = f"Please clean and canonicalize this prescription text to plain readable form (medication names, dosages, frequencies):\n\n{text}"
            params = {"max_tokens": 256}
            resp = self.client.generate(prompt=prompt, params=params)
            if isinstance(resp, dict) and resp.get("results"):
                return True, resp["results"][0].get("generated_text", "").strip()
            return True, str(resp)
        except Exception as e:
            return False, f"Granite correction failed: {e}"

    def chat(self, prompt_text: str) -> Tuple[bool, str]:
        if not self.client:
            ok, msg = self.init_client()
            if not ok:
                if "google" in prompt_text.lower() and SERPAPI_KEY:
                    return True, "[Stub] SerpAPI available - use Assistant with SerpAPI."
                return False, "Granite not configured in environment. Provide IBM credentials & SDK for real Granite."
        try:
            params = {"max_tokens": 300}
            resp = self.client.generate(prompt=prompt_text, params=params)
            if isinstance(resp, dict) and resp.get("results"):
                return True, resp["results"][0].get("generated_text", "").strip()
            return True, str(resp)
        except Exception as e:
            return False, f"Granite chat failed: {e}"

granite = GraniteSideCharacter()

# --------------------------
# Utilities and DB
# --------------------------
def select_device() -> str:
    try:
        if FORCE_CPU:
            return "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE_STR = select_device()

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            weight REAL,
            medications TEXT,
            interactions TEXT,
            created_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            message TEXT,
            created_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS consultancies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            specialization TEXT,
            phone TEXT,
            country TEXT,
            notes TEXT,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def log_db(conn: sqlite3.Connection, level: str, message: str):
    try:
        c = conn.cursor()
        c.execute("INSERT INTO logs (level, message, created_at) VALUES (?,?,?)", (level, message, datetime.now()))
        conn.commit()
    except Exception:
        pass

# --------------------------
# Drug DB & helpers
# --------------------------
DRUG_DATABASE = {
    "paracetamol": {
        "generic_name": "Acetaminophen",
        "category": "Analgesic/Antipyretic",
        "interactions": ["warfarin", "alcohol"],
        "contraindications": ["liver_disease"],
        "timing": ["after_meals"],
        "frequency": "every_6_hours",
        "max_daily": 4000,
        "age_groups": {
            "child": {"min_dose": 10, "max_dose": 15, "unit": "mg/kg"},
            "adult": {"min_dose": 500, "max_dose": 1000, "unit": "mg"},
            "elderly": {"min_dose": 325, "max_dose": 650, "unit": "mg"},
        },
    },
    "ibuprofen": {
        "generic_name": "Ibuprofen",
        "category": "NSAID",
        "interactions": ["warfarin", "lithium", "ace_inhibitors"],
        "contraindications": ["kidney_disease", "heart_disease", "stomach_ulcers"],
        "timing": ["after_meals"],
        "frequency": "every_8_hours",
        "max_daily": 3200,
        "age_groups": {
            "child": {"min_dose": 5, "max_dose": 10, "unit": "mg/kg"},
            "adult": {"min_dose": 200, "max_dose": 800, "unit": "mg"},
            "elderly": {"min_dose": 200, "max_dose": 400, "unit": "mg"},
        },
    },
    "augmentin": {
        "generic_name": "Co-amoxiclav",
        "category": "Antibiotic",
        "interactions": ["allopurinol", "warfarin"],
        "contraindications": ["penicillin_allergy"],
        "timing": ["before_meals"],
        "frequency": "every_12_hours",
        "max_daily": 3000,
        "age_groups": {
            "child": {"min_dose": 25, "max_dose": 45, "unit": "mg/kg"},
            "adult": {"min_dose": 625, "max_dose": 1000, "unit": "mg"},
            "elderly": {"min_dose": 625, "max_dose": 1000, "unit": "mg"},
        },
    },
    "enzoflam": {
        "generic_name": "Diclofenac",
        "category": "Analgesic",
        "interactions": ["warfarin", "diuretics"],
        "contraindications": ["asthma", "stomach_ulcers"],
        "timing": ["after_meals"],
        "frequency": "every_8_hours",
        "max_daily": 150,
        "age_groups": {
            "adult": {"min_dose": 50, "max_dose": 50, "unit": "mg"},
        },
    },
    "pan d": {
        "generic_name": "Pantoprazole",
        "category": "Antacid",
        "interactions": [],
        "contraindications": ["liver_disease"],
        "timing": ["before_breakfast"],
        "frequency": "once_daily",
        "max_daily": 40,
        "age_groups": {
            "adult": {"min_dose": 40, "max_dose": 40, "unit": "mg"},
        },
    }
}

ALTERNATIVES_DATABASE = {
    "paracetamol": ["ibuprofen", "aspirin", "naproxen"],
    "ibuprofen": ["paracetamol", "naproxen", "diclofenac"],
    "augmentin": ["amoxicillin", "cefixime", "azithromycin"],
    "enzoflam": ["paracetamol", "ibuprofen", "nimesulide"],
    "pan d": ["omeprazole", "rabeprazole"],
}

def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_med_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    corrections = {
        r"arugmenturi|augmenturi|augmentin[n]?": "augmentin",
        r"enziflarm|enziflarn|enzoflarm": "enzoflam",
        r"pand400mg|pand 40mg|pan d 40mg|panda": "pan d",
        r"tal\.?|tab\.?": "tab",
        r"pan ?d": "pan d",
    }
    for pat, rep in corrections.items():
        name = re.sub(pat, rep, name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# --------------------------
# OCR: Donut (HF), EasyOCR, Tesseract fallbacks
# --------------------------

# We'll cache heavy model initializations using streamlit.cache_resource
@st.cache_resource(show_spinner=False)
def init_easyocr_reader(device_str: str) -> Tuple[Optional[Any], str]:
    """
    Initialize EasyOCR reader and return (reader, msg)
    device_str: 'cuda' or 'cpu'
    """
    if not EASYOCR_AVAILABLE:
        return None, "EasyOCR not installed."
    try:
        gpu_flag = (device_str == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available())
        # languages: english is default; you can add more as needed; easyocr supports many languages
        reader = easyocr.Reader(['en'], gpu=gpu_flag)
        return reader, f"EasyOCR initialized (gpu={gpu_flag})."
    except Exception as e:
        return None, f"EasyOCR init failed: {e}"

@st.cache_resource(show_spinner=False)
def load_models(hf_token: Optional[str], device_str: str) -> Dict[str, Any]:
    """
    Attempt to load Donut (HF) and InferenceClient if HF libs and token available.
    Always also attempt to init EasyOCR (if available) so fallback is fast.
    Returns dict with keys:
      - processor (DonutProcessor or None)
      - ocr_model (VisionEncoderDecoderModel or None)
      - client (InferenceClient or None)
      - easy_reader (EasyOCR reader or None)
      - load_error (diagnostic messages)
    """
    resources = {"processor": None, "ocr_model": None, "client": None, "easy_reader": None, "load_error": None}
    msgs = []
    # Try HF Donut & client
    if HF_AVAILABLE and TRANSFORMERS_AVAILABLE:
        try:
            if hf_token:
                try:
                    login(token=hf_token)
                    msgs.append("HF login attempted.")
                except Exception as e:
                    msgs.append(f"HF login warning: {e}")
            # Load Donut
            try:
                processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME, use_auth_token=hf_token)
                ocr_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME, use_auth_token=hf_token)
                # Move to device if possible
                if device_str == "cuda" and TORCH_AVAILABLE:
                    try:
                        ocr_model.to("cuda")
                        msgs.append("Donut OCR loaded on CUDA.")
                    except Exception as e:
                        msgs.append(f"Donut->CUDA failed: {e}; using CPU.")
                        try:
                            ocr_model.to("cpu")
                        except Exception:
                            pass
                else:
                    try:
                        ocr_model.to("cpu")
                    except Exception:
                        pass
                resources["processor"] = processor
                resources["ocr_model"] = ocr_model
                msgs.append("Donut OCR model loaded.")
            except Exception as e:
                msgs.append(f"Donut OCR load failed: {e}")
            # Init InferenceClient if token present
            try:
                if hf_token:
                    client = InferenceClient(api_key=hf_token)
                    resources["client"] = client
                    msgs.append("HF InferenceClient ready.")
                else:
                    msgs.append("HF_TOKEN not provided; InferenceClient disabled.")
            except Exception as e:
                msgs.append(f"InferenceClient init failed: {e}")
        except Exception as e:
            msgs.append(f"Unexpected HF load error: {e}")
    else:
        msgs.append("HF libs not available or torch/transformers import failed.")

    # EasyOCR init (fast fallback)
    try:
        reader, rmsg = init_easyocr_reader(device_str)
        resources["easy_reader"] = reader
        msgs.append(rmsg)
    except Exception as e:
        msgs.append(f"EasyOCR init exception: {e}")

    resources["load_error"] = "\n".join(msgs)
    return resources

def extract_text_with_donut(image: Image.Image, processor, ocr_model, device_str: str) -> Tuple[str, str]:
    """
    Run Donut OCR and return (extracted_text, engine_used)
    engine_used: "donut" or an error indicator
    """
    if processor is None or ocr_model is None:
        return "❌ OCR model not loaded or unavailable.", "donut_unavailable"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Donut expects pixel_values via processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        if TORCH_AVAILABLE and device_str == "cuda" and torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
        else:
            pixel_values = pixel_values.to("cpu")
        task_prompt = "<s_ocr>"
        decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids
        if TORCH_AVAILABLE and device_str == "cuda" and torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.to("cuda")
        else:
            decoder_input_ids = decoder_input_ids.to("cpu")
        # Generate text
        with torch.no_grad():
            generated_ids = ocr_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=1024,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_text(text), "donut"
    except Exception as e:
        return f"❌ OCR error (Donut): {e}", "donut_error"

def extract_text_with_easyocr(image: Image.Image, reader, device_str: str) -> Tuple[str, str]:
    """
    Run EasyOCR to extract text. Return (text, engine_used)
    """
    if reader is None:
        return "❌ EasyOCR reader not initialized.", "easyocr_unavailable"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert image to numpy array in RGB
        img_np = np.array(image)
        # EasyOCR returns list of (bbox, text, conf)
        results = reader.readtext(img_np, detail=0)  # detail=0 returns text strings only
        text = "\n".join(results)
        return clean_text(text), "easyocr"
    except Exception as e:
        return f"❌ OCR error (EasyOCR): {e}", "easyocr_error"

def extract_text_with_tesseract(image: Image.Image) -> Tuple[str, str]:
    """
    Run pytesseract (Tesseract) as final fallback. Return (text, engine_used)
    """
    if not PYTESSERACT_AVAILABLE:
        return "❌ pytesseract not installed.", "tesseract_unavailable"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Optionally enhance contrast or convert to grayscale for Tesseract
        gray = ImageOps.grayscale(image)
        # You may tune pytesseract config (oem, psm) for prescriptions. Using default safe config here.
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(gray, config=config)
        return clean_text(text), "tesseract"
    except Exception as e:
        return f"❌ OCR error (Tesseract): {e}", "tesseract_error"

# --------------------------
# NLP extraction helpers
# --------------------------
def extract_medications_via_inference(text: str, client) -> List[Dict[str, Any]]:
    meds = []
    if client is None:
        return meds
    try:
        results = client.token_classification(text, model=NER_MODEL_NAME)
        for ent in results:
            word = ent.get("word", "").strip()
            score = ent.get("score", 0)
            label = ent.get("entity_group", ent.get("entity_label", ""))
            if not word or len(word) <= 2:
                continue
            if score < 0.55:
                continue
            if label.upper() in ("DRUG", "CHEMICAL", "MEDICATION", "SUBSTANCE"):
                meds.append({"name": normalize_med_name(word), "confidence": float(score), "type": label.upper()})
        dedup = {}
        for m in meds:
            k = m["name"]
            if k not in dedup or m["confidence"] > dedup[k]["confidence"]:
                dedup[k] = m
        return list(dedup.values())
    except Exception:
        return []

def heuristic_med_extraction(text: str) -> List[Dict[str, Any]]:
    text = clean_text(text)
    meds = []
    # Matches words of length >=3 with optional mg/dosage attached
    tokens = re.findall(r"([A-Za-z]{3,}(?:[\s\-]*\d{1,4}\s*(?:mg|mcg|g)?)?)", text, flags=re.IGNORECASE)
    for t in tokens:
        name_only = re.sub(r"\d+.*", "", t).strip()
        name = normalize_med_name(name_only)
        if not name:
            continue
        matched = None
        for d in DRUG_DATABASE.keys():
            if d in name or name in d:
                matched = d
                break
        if matched:
            meds.append({"name": matched, "confidence": 0.95, "type": "HEURISTIC_DB"})
        else:
            meds.append({"name": name, "confidence": 0.60, "type": "HEURISTIC"})
    dedup = {}
    for m in meds:
        k = m["name"]
        if k not in dedup or m["confidence"] > dedup[k]["confidence"]:
            dedup[k] = m
    return list(dedup.values())

def extract_medications(text: str, client) -> List[Dict[str, Any]]:
    text = clean_text(text)
    meds = []
    # Primary extraction via HF NER if available
    if client is not None:
        try:
            meds = extract_medications_via_inference(text, client)
        except Exception:
            meds = []
    # Fallback heuristic
    if not meds:
        meds = heuristic_med_extraction(text)
    # Granite side-character suggestion/correction (non-blocking)
    try:
        ok, corr = granite.correct_text(text)
        if ok and corr:
            corr_meds = heuristic_med_extraction(corr)
            existing = {m["name"]: m for m in meds}
            for cm in corr_meds:
                if (cm["name"] not in existing) or (cm["confidence"] > existing[cm["name"]]["confidence"]):
                    existing[cm["name"]] = cm
            meds = list(existing.values())
    except Exception:
        pass
    return meds

# --------------------------
# Interaction, dosage & scheduling
# --------------------------
def detect_drug_interactions(medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interactions = []
    med_names = [m["name"].lower() for m in medications]
    for i in range(len(med_names)):
        for j in range(i+1, len(med_names)):
            a = med_names[i]
            b = med_names[j]
            for drug_name, drug_info in DRUG_DATABASE.items():
                if drug_name in a or a in drug_name:
                    for inter in drug_info.get("interactions", []):
                        if inter in b or b in inter:
                            interactions.append({
                                "drug1": a,
                                "drug2": b,
                                "severity": "HIGH",
                                "description": f"Potential interaction between {a} and {b}",
                            })
    return interactions

def get_age_group(age: int) -> str:
    if age < 18:
        return "child"
    if age >= 65:
        return "elderly"
    return "adult"

def get_age_appropriate_dosage(medication: str, age: int, weight: Optional[float] = None) -> Optional[Dict[str, Any]]:
    med_lower = medication.lower()
    age_group = get_age_group(age)
    for drug_name, drug_info in DRUG_DATABASE.items():
        if drug_name in med_lower or med_lower in drug_name:
            dosage_info = drug_info.get("age_groups", {}).get(age_group)
            if not dosage_info:
                return None
            rec = {
                "medication": medication,
                "age_group": age_group,
                "min_dose": dosage_info.get("min_dose"),
                "max_dose": dosage_info.get("max_dose"),
                "unit": dosage_info.get("unit"),
                "frequency": drug_info.get("frequency"),
                "timing": drug_info.get("timing"),
                "max_daily": drug_info.get("max_daily"),
            }
            if age_group == "child" and weight and "mg/kg" in (dosage_info.get("unit") or ""):
                rec["calculated_min"] = dosage_info["min_dose"] * weight
                rec["calculated_max"] = dosage_info["max_dose"] * weight
            return rec
    return None

def suggest_alternatives(medication: str) -> List[str]:
    med_lower = medication.lower()
    for drug_name, alts in ALTERNATIVES_DATABASE.items():
        if drug_name in med_lower or med_lower in drug_name:
            return alts
    return []

def generate_schedule(medications: List[Dict[str, Any]], first_dt: datetime, days: int, patient_age: int, patient_weight: Optional[float] = None) -> List[Dict[str, Any]]:
    schedule = []
    hours_map = {
        "every_4_hours": 4,
        "every_6_hours": 6,
        "every_8_hours": 8,
        "every_12_hours": 12,
        "once_daily": 24,
        "twice_daily": 12,
        "three_times_daily": 8,
    }
    for med in medications:
        rec = get_age_appropriate_dosage(med["name"], patient_age, patient_weight)
        freq_str = rec.get("frequency") if rec else med.get("frequency", "every_8_hours")
        gap = hours_map.get(freq_str, 8)
        count_per_day = max(1, 24 // gap)
        timing_guidance = rec.get("timing") if rec else []
        for d in range(days):
            for t in range(count_per_day):
                dose_time = first_dt + timedelta(days=d, hours=t * gap)
                schedule.append({
                    "medication": med["name"],
                    "time": dose_time,
                    "timing_guidance": ", ".join(timing_guidance) if isinstance(timing_guidance, list) else (timing_guidance or ""),
                    "taken": False
                })
    return schedule

# --------------------------
# SerpAPI helper
# --------------------------
def google_search_snippet(query: str) -> str:
    if not SERPAPI_KEY:
        return "SerpAPI key not set. Set SERPAPI_KEY in .env to enable live snippets."
    try:
        r = requests.get("https://serpapi.com/search", params={"q": query, "api_key": SERPAPI_KEY, "hl": "en"})
        data = r.json()
        if "organic_results" in data and data["organic_results"]:
            return data["organic_results"][0].get("snippet") or data["organic_results"][0].get("title") or "No snippet."
        return "No results."
    except Exception as e:
        return f"Search error: {e}"

# --------------------------
# UI and main
# --------------------------
def load_css():
    st.markdown("""
    <style>
    .main-header {padding:12px;border-radius:12px;background:linear-gradient(90deg,#0f172a,#0b1220);color:#fff}
    .medicine-card {padding:10px;border-radius:8px;background:#0b1220;border:1px solid #0ea5a4;margin-bottom:8px}
    .interaction-warning {padding:10px;border-radius:8px;background:#3f1f1f;color:#fff;border-left:6px solid #ff4d4f}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def sample_data():
    return {}

def main():
    conn = init_db(DB_PATH)
    load_css()
    st.title("🏥 Smart Medicine Assistant ")
    st.caption("Donut OCR (HF) primary; EasyOCR & Tesseract fallback. Will auto-select best OCR and use GPU when available.")

    # Sidebar
    with st.sidebar:
        st.header("👤 Patient Profile")
        st.session_state.setdefault("patient_name", "")
        st.session_state.setdefault("patient_age", 30)
        st.session_state.setdefault("patient_weight", 70.0)
        st.session_state["patient_name"] = st.text_input("Name", value=st.session_state["patient_name"])
        st.session_state["patient_age"] = st.slider("Age", min_value=1, max_value=120, value=st.session_state["patient_age"])
        st.session_state["patient_weight"] = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, value=st.session_state["patient_weight"])

        st.markdown("---")
        st.header("🔧 App Status & Controls")
        st.write(f"Device: **{DEVICE_STR.upper()}**")

        ok, msg = check_hf_token()
        if ok:
            st.success("HF token: OK")
        else:
            st.warning("HF token missing or invalid")
            st.info(msg)

        st.write("HF libs installed:", HF_AVAILABLE)
        st.write("Torch available:", TORCH_AVAILABLE)
        st.write("EasyOCR installed:", EASYOCR_AVAILABLE)
        st.write("pytesseract installed:", PYTESSERACT_AVAILABLE)

        # Granite info
        g_ok_msg = "Granite SDK installed." if granite.sdk_available else "Granite SDK not installed (using stub)."
        st.write("Granite side-character:", g_ok_msg)
        if granite.sdk_available:
            st.write("IBM credentials present:", bool(os.environ.get("IBM_API_KEY") and os.environ.get("IBM_PROJECT_ID")))
        else:
            st.info("Granite is present in code as a safe stub; no IBM credentials required.")

        if TWILIO_AVAILABLE and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
            st.success("SMS alerts enabled (Twilio).")
        else:
            st.info("SMS disabled. Set Twilio env vars & install twilio to enable.")

        if st.button("🔁 Clear App Data"):
            for k in ["medications", "medication_schedule", "notifications", "chat_history", "last_extracted_text", "last_used_engine"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

        if st.button("🔄 Save Prescription"):
            try:
                meds = st.session_state.get("medications", [])
                c = conn.cursor()
                c.execute("INSERT INTO prescriptions (patient_name, age, weight, medications, interactions, created_at) VALUES (?,?,?,?,?,?)",
                          (st.session_state.get("patient_name",""), st.session_state.get("patient_age",0), st.session_state.get("patient_weight",0.0),
                           json.dumps(meds, default=str), json.dumps(detect_drug_interactions(meds)), datetime.now()))
                conn.commit()
                st.success("Saved to local DB.")
            except Exception as e:
                st.error("Save failed: " + str(e))
                log_db(conn, "ERROR", f"Save failed: {str(e)}")

    # Model & service status
    models = {"processor": None, "ocr_model": None, "client": None, "easy_reader": None, "load_error": None}
    with st.expander("Model & Service Status (click)"):
        if not HF_AVAILABLE:
            st.warning("HF torch/transformers not installed; Donut/Inference disabled.")
        else:
            with st.spinner("Loading HF models (if HF_TOKEN present)..."):
                models = load_models(HF_TOKEN, DEVICE_STR)
            if models.get("load_error"):
                st.info(models["load_error"])
        st.write("Device:", DEVICE_STR)
        st.write("HF token present:", bool(HF_TOKEN))
        st.write("SerpAPI configured:", bool(SERPAPI_KEY))

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📷 OCR & Extract",
        "💊 Medication Manager",
        "⚠️ Safety Check",
        "⏰ Schedule",
        "📊 Dashboard",
        "🤖 Assistant (Granite side-character)"
    ])

    st.session_state.setdefault("medications", [])
    st.session_state.setdefault("medication_schedule", [])
    st.session_state.setdefault("notifications", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_used_engine", None)

    # Tab 1 - OCR & extraction
    with tab1:
        st.header("📷 OCR & Medication Extraction")
        col1, col2 = st.columns([2,1])
        with col1:
            uploaded = st.file_uploader("Upload prescription (PNG/JPG). Or paste text below:", type=["png","jpg","jpeg"])
            manual_text = st.text_area("Or paste raw prescription text:", value="", height=140)
            image = None
            if uploaded:
                try:
                    image = Image.open(uploaded)
                    st.image(image, caption="Prescription preview", use_container_width=True)
                except Exception as e:
                    st.error("Image open failed: " + str(e))
                    image = None

            if st.button("🔍 Run OCR & Extract"):
                extracted = ""
                engine_used = None
                try:
                    # Preference: Donut -> EasyOCR -> Tesseract -> Manual
                    if image:
                        # Try Donut OCR first if models loaded
                        if models.get("processor") and models.get("ocr_model") and HF_AVAILABLE and HF_TOKEN and models.get("processor") is not None:
                            with st.spinner("Running Donut OCR (Hugging Face) ..."):
                                extracted, engine_used = extract_text_with_donut(image, models.get("processor"), models.get("ocr_model"), DEVICE_STR)
                                # If Donut returned an error string starting with ❌, treat as failure
                                if extracted.startswith("❌"):
                                    # fallback to EasyOCR
                                    extracted = ""
                                    engine_used = None
                                else:
                                    st.session_state["last_used_engine"] = engine_used
                                    st.session_state["last_extracted_text"] = extracted
                        # If Donut not used or failed, try EasyOCR
                        if not engine_used and models.get("easy_reader"):
                            with st.spinner("Running EasyOCR fallback ..."):
                                extracted, engine_used = extract_text_with_easyocr(image, models.get("easy_reader"), DEVICE_STR)
                                if extracted.startswith("❌"):
                                    extracted = ""
                                    engine_used = None
                                else:
                                    st.session_state["last_used_engine"] = engine_used
                                    st.session_state["last_extracted_text"] = extracted
                        # If still nothing, try Tesseract
                        if not engine_used:
                            if PYTESSERACT_AVAILABLE:
                                with st.spinner("Running Tesseract fallback ..."):
                                    extracted, engine_used = extract_text_with_tesseract(image)
                                    if extracted.startswith("❌"):
                                        extracted = ""
                                        engine_used = None
                                    else:
                                        st.session_state["last_used_engine"] = engine_used
                                        st.session_state["last_extracted_text"] = extracted
                            else:
                                # no OCR available
                                engine_used = None
                                extracted = ""
                        # If none produced text, fallback to manual text
                        if not extracted:
                            if manual_text and manual_text.strip():
                                extracted = manual_text
                                engine_used = "manual_provided"
                                st.session_state["last_used_engine"] = engine_used
                                st.session_state["last_extracted_text"] = extracted
                            else:
                                # Provide user-friendly message, not an error flood
                                extracted = "⚠️ OCR could not extract text automatically. Provide manual text or try a clearer image."
                                engine_used = "none"
                                st.session_state["last_used_engine"] = engine_used
                                st.session_state["last_extracted_text"] = extracted
                    else:
                        # No image — use manual text
                        extracted = manual_text or ""
                        engine_used = "manual"
                        st.session_state["last_used_engine"] = engine_used
                        st.session_state["last_extracted_text"] = extracted

                    # Show the engine used as an info box
                    if engine_used == "donut":
                        st.success("OCR engine: Donut (Hugging Face).")
                    elif engine_used == "easyocr":
                        st.info("OCR engine: EasyOCR (fallback).")
                    elif engine_used == "tesseract":
                        st.info("OCR engine: Tesseract (final fallback).")
                    elif engine_used in ("manual", "manual_provided"):
                        st.info("Using manual text provided.")
                    else:
                        st.warning("No automatic OCR succeeded; using manual text or try a different image.")

                    st.text_area("Extracted/OCR text (read-only)", value=extracted, height=160, disabled=True)

                    meds = []
                    try:
                        # If extracted text looks like an error message, skip auto extraction
                        if extracted and ("❌" not in extracted and "⚠️" not in extracted):
                            meds = extract_medications(extracted, models.get("client"))
                        if not meds and manual_text:
                            meds = extract_medications(manual_text, models.get("client"))
                    except Exception as e:
                        st.error("Extraction error: " + str(e))
                        log_db(conn, "ERROR", f"Extraction error: {str(e)}")
                        meds = []

                    if meds:
                        existing = {m["name"]: m for m in st.session_state.get("medications", [])}
                        for m in meds:
                            nm = {"name": normalize_med_name(m.get("name","")), "confidence": m.get("confidence", 0.6), "type": m.get("type","AUTO")}
                            existing[nm["name"]] = nm
                        st.session_state["medications"] = list(existing.values())
                        st.success(f"Extracted {len(meds)} medication(s).")
                    else:
                        st.warning("No medications found — try manual entry or improve the image/text quality.")
                except Exception as e:
                    st.error("OCR/extract pipeline error: " + str(e))
                    log_db(conn, "ERROR", f"OCR pipeline error: {str(e)}")
                    traceback.print_exc()

        with col2:
            st.markdown("#### Manual Add / Quick Fix")
            med_name = st.text_input("Medication name (manual)")
            med_dose = st.text_input("Dosage (optional e.g., 500mg)")
            med_freq = st.selectbox("Frequency (optional)", ["", "once_daily", "twice_daily", "every_4_hours", "every_6_hours", "every_8_hours", "every_12_hours"])
            if st.button("➕ Add Medication"):
                if med_name:
                    nm = normalize_med_name(med_name)
                    new_med = {"name": nm, "dosage": med_dose, "frequency": med_freq, "confidence": 1.0, "type": "MANUAL"}
                    existing = {m["name"]: m for m in st.session_state.get("medications", [])}
                    existing[nm] = new_med
                    st.session_state["medications"] = list(existing.values())
                    st.success(f"Added {nm.title()}")

    # Tab 2 - Medication Manager
    with tab2:
        st.header("💊 Medication Manager")
        meds = st.session_state.get("medications", [])
        if meds:
            for idx, med in enumerate(meds):
                with st.expander(f"{med['name'].title()}  •  {med.get('type','')}", expanded=False):
                    a,b,c = st.columns([3,3,1])
                    with a:
                        st.write("**Name:**", med["name"].title())
                        if med.get("dosage"):
                            st.write("**Dosage:**", med["dosage"])
                        st.write("**Confidence:**", f"{med.get('confidence',0.0):.2f}")
                    with b:
                        rec = get_age_appropriate_dosage(med["name"], st.session_state.get("patient_age",30), st.session_state.get("patient_weight",70.0))
                        if rec:
                            st.write("**Recommended Dosage**")
                            st.write(f"{rec.get('min_dose')} - {rec.get('max_dose')} {rec.get('unit')}")
                            if rec.get("calculated_min"):
                                st.write(f"For {st.session_state.get('patient_weight')}kg: {rec['calculated_min']:.1f} - {rec['calculated_max']:.1f} mg")
                            st.write("When:", rec.get("timing") or rec.get("when_to_take",""))
                    with c:
                        if st.button("Remove", key=f"rm_{idx}"):
                            st.session_state["medications"].pop(idx)
                            st.experimental_rerun()
        else:
            st.info("No medications in list yet.")

    # Tab 3 - Safety Check
    with tab3:
        st.header("⚠️ Safety & Interaction")
        meds = st.session_state.get("medications", [])
        if meds:
            interactions = detect_drug_interactions(meds)
            if interactions:
                st.error(f"{len(interactions)} potential interaction(s) detected.")
                for inter in interactions:
                    st.markdown(f"- **{inter['drug1'].title()} ↔ {inter['drug2'].title()}** — {inter['description']}")
                if TWILIO_AVAILABLE and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
                    phone = st.text_input("Phone for SMS alert (E.164)", value="")
                    if st.button("Send SMS Alert"):
                        if phone:
                            try:
                                client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
                                client.messages.create(body="Potential drug interaction detected. Please consult provider.", from_=TWILIO_FROM, to=phone)
                                st.success("SMS sent.")
                            except Exception as e:
                                st.warning("SMS failed: " + str(e))
            else:
                st.success("✅ No major interactions detected.")
            st.markdown("#### Alternatives")
            for med in meds:
                alts = suggest_alternatives(med["name"])
                if alts:
                    with st.expander(f"Alternatives for {med['name'].title()}"):
                        for a in alts:
                            st.write("• " + a.title())
                        st.warning("Consult provider before switching.")
        else:
            st.info("Add medications to analyze.")

    # Tab 4 - Schedule
    with tab4:
        st.header("⏰ Schedule & Reminders")
        meds = st.session_state.get("medications", [])
        if meds:
            left, right = st.columns([2,1])
            with left:
                start_time = st.time_input("First dose time", value=datetime.now().time())
                days = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
            with right:
                default_freq = st.selectbox("Default frequency (if unknown)", ["every_8_hours","every_6_hours","every_12_hours","once_daily"])
            if st.button("Create schedule"):
                first_dt = datetime.combine(datetime.now().date(), start_time)
                schedule = generate_schedule(st.session_state["medications"], first_dt, int(days), st.session_state.get("patient_age",30), st.session_state.get("patient_weight",70.0))
                existing = st.session_state.get("medication_schedule", [])
                ex_keys = {(s['medication'], s['time']) for s in existing}
                for s in schedule:
                    if (s["medication"], s["time"]) not in ex_keys:
                        existing.append(s)
                st.session_state["medication_schedule"] = existing
                st.success("Schedule created.")
            now = datetime.now()
            upcoming = [s for s in st.session_state.get("medication_schedule", []) if s["time"] >= now and s["time"] <= now + timedelta(days=7)]
            if upcoming:
                df = pd.DataFrame(upcoming)
                df["time_str"] = df["time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if isinstance(x, datetime) else str(x))
                st.dataframe(df[["medication","time_str","timing_guidance","taken"]].rename(columns={"medication":"Medication","time_str":"Time","timing_guidance":"When","taken":"Taken"}))
            else:
                st.info("No upcoming scheduled doses.")
        else:
            st.info("No meds to schedule.")

    # Tab 5 - Dashboard
    with tab5:
        st.header("📊 Dashboard")
        meds = st.session_state.get("medications", [])
        if meds:
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("Total medications", len(meds))
            with c2:
                interactions = detect_drug_interactions(meds)
                st.metric("Interactions", len(interactions), delta="High" if interactions else "Safe")
            with c3:
                sched = st.session_state.get("medication_schedule", [])
                if sched:
                    taken = sum(1 for s in sched if s.get("taken"))
                    total = len(sched)
                    compliance = (taken/total)*100 if total>0 else 0
                    st.metric("Compliance", f"{compliance:.1f}%")
                else:
                    st.metric("Compliance", "No data")
            with c4:
                upcoming = [s for s in st.session_state.get("medication_schedule", []) if (not s.get("taken")) and s.get("time") >= datetime.now()]
                st.metric("Upcoming doses", len(upcoming))

            categories = {}
            for m in meds:
                found = False
                for key, info in DRUG_DATABASE.items():
                    if key in m["name"].lower() or m["name"].lower() in key:
                        categories[info.get("category","Other")] = categories.get(info.get("category","Other"), 0) + 1
                        found = True
                        break
                if not found:
                    categories["Other"] = categories.get("Other", 0) + 1
            if categories:
                fig = px.pie(values=list(categories.values()), names=list(categories.keys()), title="Medication Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No medications to display.")

    # Tab 6 - Assistant (Granite side-character)
    with tab6:
        st.header("🤖 Assistant (Granite side-character + SerpAPI fallback)")
        st.markdown("Granite is present in code as a side-character (safe stub). If you install IBM SDK and provide IBM_API_KEY & IBM_PROJECT_ID it will attempt real Granite calls.")
        user_q = st.text_input("Ask assistant (medications, dosages, interactions, or request correction):")
        correction = st.checkbox("Request correction of last extracted text (if available)")
        if st.button("Send to Assistant"):
            st.session_state.setdefault("chat_history", [])
            last_text = st.session_state.get("last_extracted_text", "")
            if correction and last_text:
                ok, resp = granite.correct_text(last_text)
                if ok:
                    st.session_state["chat_history"].append(("You (correction)", "Correct last extracted text"))
                    st.session_state["chat_history"].append(("Assistant (Granite)", resp))
                else:
                    st.session_state["chat_history"].append(("Assistant", resp))
            else:
                if user_q.strip() == "":
                    st.warning("Type a question or request.")
                else:
                    ok, resp = granite.chat(user_q)
                    if ok:
                        st.session_state["chat_history"].append(("You", user_q))
                        st.session_state["chat_history"].append(("Assistant (Granite)", resp))
                    else:
                        if SERPAPI_KEY:
                            snippet = google_search_snippet(user_q)
                            st.session_state["chat_history"].append(("You", user_q))
                            st.session_state["chat_history"].append(("Assistant (Search)", snippet))
                        else:
                            st.session_state["chat_history"].append(("Assistant", resp))
        for role, msg in st.session_state.get("chat_history", []):
            if role.startswith("Assistant"):
                st.markdown(f"**{role}:** {msg}")
            else:
                st.markdown(f"**{role}:** {msg}")

    # Notifications quick-run (passive)
    if "last_reminder_check" not in st.session_state:
        st.session_state["last_reminder_check"] = datetime.now()
    now = datetime.now()
    if (now - st.session_state["last_reminder_check"]).seconds >= 30:
        st.session_state["last_reminder_check"] = now
        due = []
        for item in st.session_state.get("medication_schedule", []):
            if not item.get("taken") and item.get("time") <= now <= item.get("time") + timedelta(minutes=30):
                due.append(item)
        for d in due:
            nid = f"{d['medication']}_{d['time']}"
            existing_ids = [n.get("id") for n in st.session_state.get("notifications",[])]
            if nid not in existing_ids:
                st.session_state["notifications"].append({"id":nid, "medication":d["medication"], "time":d["time"], "message":f"Time to take {d['medication']}", "created": now})

    if st.session_state.get("notifications"):
        with st.expander("🔔 Active Notifications"):
            for n in list(st.session_state["notifications"]):
                st.write(f"- {n['message']} at {n['time'].strftime('%Y-%m-%d %H:%M')}")
                if st.button(f"Mark {n['id']} handled", key=f"notif_{n['id']}"):
                    st.session_state["notifications"] = [x for x in st.session_state["notifications"] if x["id"] != n["id"]]
                    st.success("Handled.")

    # consultancies / emergency
    st.markdown("---")
    with st.expander("🚨 Emergency / Consultancies"):
        st.write("Emergency numbers: India 108/112, USA 911, UK 999. In case of emergency call local services.")
        c_name = st.text_input("Add consultancy - Name")
        c_spec = st.text_input("Specialization")
        c_phone = st.text_input("Phone (international)")
        c_country = st.text_input("Country")
        c_notes = st.text_area("Notes / availability")
        if st.button("Add Consultancy"):
            try:
                cur = conn.cursor()
                cur.execute("INSERT INTO consultancies (name, specialization, phone, country, notes, created_at) VALUES (?,?,?,?,?,?)",
                            (c_name, c_spec, c_phone, c_country, c_notes, datetime.now()))
                conn.commit()
                st.success("Added.")
            except Exception as e:
                st.error("Failed to add: " + str(e))
                log_db(conn, "ERROR", f"Consult add failed: {str(e)}")
        cur = conn.cursor()
        cur.execute("SELECT id,name,specialization,phone,country,notes FROM consultancies ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()
        if rows:
            dfc = pd.DataFrame(rows, columns=["id","name","specialization","phone","country","notes"])
            st.dataframe(dfc[["name","specialization","phone","country"]])
        else:
            st.info("No consultancies added yet.")

    # footer export
    st.markdown("---")
    left, right = st.columns([1,3])
    with left:
        if st.button("💾 Download JSON Report"):
            rpt = {"patient": {"name": st.session_state.get("patient_name",""), "age": st.session_state.get("patient_age",0), "weight": st.session_state.get("patient_weight",0.0)},
                   "medications": st.session_state.get("medications", []),
                   "schedule": st.session_state.get("medication_schedule", []),
                   "generated_on": datetime.now().isoformat(),
                   "last_used_engine": st.session_state.get("last_used_engine", None)}
            st.download_button("Download JSON", data=json.dumps(rpt, default=str, indent=2), file_name=f"med_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
    with right:
        st.markdown("_Demo only. Not a substitute for medical advice._")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Avoid exposing stack traces to UI excessively; log and show friendly message
        try:
            st.error("Unexpected error occurred. Check logs.")
        except Exception:
            pass
        traceback.print_exc()
streamlit run "C:\Users\Rishith varma\medverify_env\medical_ui_new.py" 