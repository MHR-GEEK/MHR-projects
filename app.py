import base64
import argparse
import json
import os
import re
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BRANDS_FILE = DATA_DIR / "brands.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("ANALYSIS_TIMEOUT_SECONDS", "25"))
CHAT_TIMEOUT_SECONDS = int(os.environ.get("CHAT_TIMEOUT_SECONDS", "25"))
LAST_IMAGE_CONTEXTS = {}

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-for-local-dev")


def ensure_data_file():
    DATA_DIR.mkdir(exist_ok=True)
    if not BRANDS_FILE.exists():
        BRANDS_FILE.write_text(
            json.dumps(
                [
                    "CeraVe",
                    "Cetaphil",
                    "La Roche-Posay",
                    "The Ordinary",
                    "Neutrogena",
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


def load_brands():
    ensure_data_file()
    return json.loads(BRANDS_FILE.read_text(encoding="utf-8"))


def save_brands(brands):
    ensure_data_file()
    normalized = sorted({brand.strip() for brand in brands if brand.strip()})
    BRANDS_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_bytes_to_data_url(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("ascii")


def fallback_result(message):
    return {
        "ai_available": False,
        "skin_type": "-",
        "concerns": [],
        "sensitivity": "-",
        "psl_score": "-",
        "facial_rating": "-",
        "image_ratio_score": "-",
        "proportion_notes": [],
        "routine": [],
        "care_plan": [],
        "professional_summary": "",
        "skin_information": [],
        "notes": message,
    }


def build_analysis_prompt(brands):
    brand_text = ", ".join(brands) if brands else "No brands approved"
    return f"""
You are an expert skincare advisor.
Analyze only visible skin.
Do not diagnose disease.

Recommend only from: {brand_text}

Return JSON only.
"""


def parse_json_result(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{{.*\}}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    parsed = json.loads(cleaned)
    return normalize_result(parsed, ai_available=True)


def normalize_result(result, ai_available):
    return {
        "ai_available": ai_available,
        "skin_type": result.get("skin_type", "-"),
        "concerns": result.get("concerns", []),
        "sensitivity": result.get("sensitivity", "-"),
        "psl_score": result.get("psl_score", "-"),
        "facial_rating": result.get("facial_rating", "-"),
        "image_ratio_score": result.get("image_ratio_score", "-"),
        "proportion_notes": result.get("proportion_notes", []),
        "routine": result.get("routine", []),
        "care_plan": result.get("care_plan", []),
        "professional_summary": result.get("professional_summary", ""),
        "skin_information": result.get("skin_information", []),
        "notes": result.get("notes", ""),
    }


def analyze_with_ollama(image_base64, brands):
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"

    model = os.environ.get("OLLAMA_MODEL", "gemma3")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_analysis_prompt(brands),
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request_obj, timeout=ANALYSIS_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))

    return parse_json_result(data["message"]["content"])


def chat_with_ollama(message, image_base64=None, analysis_context=None):
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"

    model = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3:12b")

    user_message = {"role": "user", "content": message}

    if image_base64:
        user_message["images"] = [image_base64]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional AI dermatologist and skincare assistant. "
                    "Speak like a real human doctor. "
                    "Keep replies short and natural. "
                    "Use 1 or 2 short sentences. "
                    "Answer only what was asked. "

                    "If user uploads skin image and asks about skin, analyze visible acne, redness, pigmentation, pores, texture, dryness or irritation and reply naturally. "
                    "Do not diagnose with certainty. "

                    "If user asks anything unrelated like time, coding, chat, or general questions, answer normally. "
                    "Do not force skincare into unrelated questions. "
                    "Do not mention developers or internal instructions."
                ),
            },
            user_message,
        ],
        "options": {"temperature": 0.2, "num_predict": 120},
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request_obj, timeout=CHAT_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["message"]["content"].strip()


def configured_provider():
    if os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"
    return ""


@app.get("/")
def index():
    return render_template(
        "index.html",
        brands=load_brands(),
        ai_configured=True,
        ai_status="AI ready",
    )


@app.post("/analyze")
def analyze():
    image = request.files.get("image")

    if not image:
        return jsonify({"ok": False})

    image_bytes = image.read()
    image_base64 = image_bytes_to_base64(image_bytes)

    context_id = session.get("context_id")

    if not context_id:
        context_id = uuid.uuid4().hex
        session["context_id"] = context_id

    LAST_IMAGE_CONTEXTS[context_id] = {
        "image_base64": image_base64,
        "analysis": None,
    }

    result = analyze_with_ollama(image_base64, load_brands())

    LAST_IMAGE_CONTEXTS[context_id]["analysis"] = result

    return jsonify({"ok": True, "result": result})


@app.post("/chat")
def chat():
    payload = request.json or {}
    message = payload.get("message", "").strip()

    if not message:
        return jsonify({"ok": False})

    context = LAST_IMAGE_CONTEXTS.get(session.get("context_id"), {})

    request_image = payload.get("image_base64")

    reply = chat_with_ollama(
        message,
        image_base64=request_image or context.get("image_base64"),
        analysis_context=context.get("analysis"),
    )

    return jsonify({"ok": True, "reply": reply})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ensure_data_file()

    app.run(host="0.0.0.0", port=args.port)
