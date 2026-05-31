````python
import base64
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BRANDS_FILE = DATA_DIR / "brands.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

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
        "skin_information": [],
        "routine": [],
        "care_plan": [],
        "notes": message,
    }


def build_analysis_prompt(brands):
    brand_text = ", ".join(brands) if brands else "No brands approved"

    return f"""
You are an educational skincare assistant.

Analyze only visible skincare traits.

Never leave any field empty.

Only recommend products from:
{brand_text}

Return strict JSON:

{{
  "skin_type": "oily | dry | combination | normal | unclear",
  "concerns": ["minimum 2 visible concerns"],
  "sensitivity": "low | moderate | high | unclear",
  "psl_score": "1.0 to 10.0",
  "facial_rating": "low | average | above average | high",
  "image_ratio_score": "1 to 100",
  "proportion_notes": ["minimum 3 notes"],
  "routine": [
    {{"step":"AM cleanser","recommendation":"..."}},
    {{"step":"AM moisturizer","recommendation":"..."}},
    {{"step":"AM sunscreen","recommendation":"..."}},
    {{"step":"PM cleanser","recommendation":"..."}},
    {{"step":"PM treatment","recommendation":"..."}},
    {{"step":"PM moisturizer","recommendation":"..."}}
  ],
  "notes": "minimum 2 sentences"
}}
"""


def parse_json_result(raw_text):
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        cleaned = match.group(0)

    parsed = json.loads(cleaned)
    return normalize_result(parsed, ai_available=True)


def normalize_result(result, ai_available):
    concerns = result.get("concerns")

    if not isinstance(concerns, list) or not concerns:
        concerns = [
            "visible skin texture",
            "general skincare monitoring",
        ]

    proportion_notes = result.get("proportion_notes")

    if not isinstance(proportion_notes, list) or not proportion_notes:
        proportion_notes = [
            "Visible facial structure reviewed.",
            "Lighting and angle considered.",
            "Skin texture estimated from image.",
        ]

    routine = result.get("routine")

    if not isinstance(routine, list) or not routine:
        routine = [
            {"step": "AM cleanser", "recommendation": "Use a gentle cleanser."},
            {"step": "AM moisturizer", "recommendation": "Apply lightweight moisturizer."},
            {"step": "AM sunscreen", "recommendation": "Use SPF 30+."},
            {"step": "PM cleanser", "recommendation": "Cleanse before sleep."},
            {"step": "PM treatment", "recommendation": "Use targeted treatment if needed."},
            {"step": "PM moisturizer", "recommendation": "Hydrate overnight."},
        ]

    notes = result.get("notes") or (
        "Educational skincare guidance only. See dermatologist if symptoms persist."
    )

    return {
        "ai_available": ai_available,
        "skin_type": result.get("skin_type") or "unclear",
        "concerns": concerns,
        "sensitivity": result.get("sensitivity") or "moderate",
        "psl_score": result.get("psl_score") or "6.5",
        "facial_rating": result.get("facial_rating") or "average",
        "image_ratio_score": result.get("image_ratio_score") or "70",
        "proportion_notes": proportion_notes,
        "skin_information": proportion_notes,
        "routine": routine,
        "care_plan": [notes],
        "notes": notes,
    }


def analyze_with_openai(image_data_url, brands):
    from openai import OpenAI

    client = OpenAI()
    prompt = build_analysis_prompt(brands)

    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )

    return parse_json_result(response.output_text)


def analyze_with_ollama(image_base64, brands):
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")

    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"

    model = os.environ.get("OLLAMA_MODEL", "gemma3")
    prompt = build_analysis_prompt(brands)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
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

    with urllib.request.urlopen(request_obj, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data.get("message", {}).get("content", "")
    return parse_json_result(content)


def configured_provider():
    provider = os.environ.get("AI_PROVIDER", "").strip().lower()

    if provider:
        return provider

    if os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return ""


@app.get("/")
def index():
    return render_template(
        "index.html",
        brands=load_brands(),
        ai_configured=bool(configured_provider()),
        ai_status="AI ready" if configured_provider() else "AI key missing",
    )


@app.post("/analyze")
def analyze():
    image = request.files.get("image")

    if not image or image.filename == "":
        return jsonify(
            {"ok": False, "message": "Upload a face image first"}
        ), 400

    if not allowed_file(image.filename):
        return jsonify(
            {"ok": False, "message": "Use PNG/JPG/WEBP"}
        ), 400

    image_bytes = image.read()
    provider = configured_provider()

    if not provider:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result("AI not configured"),
            }
        )

    try:
        if provider == "ollama":
            result = analyze_with_ollama(
                image_bytes_to_base64(image_bytes),
                load_brands(),
            )
        else:
            result = analyze_with_openai(
                image_bytes_to_data_url(
                    image_bytes,
                    image.mimetype or "image/png",
                ),
                load_brands(),
            )

    except Exception as exc:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(f"AI failed: {exc}"),
            }
        )

    return jsonify(
        {
            "ok": True,
            "result": result,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ensure_data_file()

    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=True,
    )
````
