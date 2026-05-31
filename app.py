import base64
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BRANDS_FILE = DATA_DIR / "brands.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)


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
        "professional_summary": "",
        "skin_information": [],
        "routine": [],
        "care_plan": [],
        "notes": message,
    }


def build_analysis_prompt(brands):
    brand_text = ", ".join(brands) if brands else "No brands approved"

    return f"""
You are a professional skincare assistant.

Analyze only visible skin from uploaded image.

Recommend only from:
{brand_text}

Return strict JSON:

{{
  "skin_type": "oily/dry/combination/normal/unclear",
  "concerns": ["max 3"],
  "sensitivity": "low/moderate/high/unclear",
  "psl_score": "1.0 to 10.0",
  "facial_rating": "low/average/above average/high/unclear",
  "image_ratio_score": "1 to 100",
  "proportion_notes": ["max 3"],

  "professional_summary": "short human skincare overview",

  "skin_information": [
    "visible texture",
    "tone and pigmentation",
    "hydration/oil balance"
  ],

  "routine": [
    {{"step":"AM cleanser","recommendation":"..."}},
    {{"step":"AM moisturizer","recommendation":"..."}},
    {{"step":"AM sunscreen","recommendation":"..."}},
    {{"step":"PM cleanser","recommendation":"..."}},
    {{"step":"PM treatment","recommendation":"..."}},
    {{"step":"PM moisturizer","recommendation":"..."}}
  ],

  "care_plan": [
    "tip 1",
    "tip 2",
    "tip 3"
  ],

  "notes":"brief dermatologist note"
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
    return {
        "ai_available": ai_available,
        "skin_type": result.get("skin_type", "-"),
        "concerns": result.get("concerns", []),
        "sensitivity": result.get("sensitivity", "-"),
        "psl_score": result.get("psl_score", "-"),
        "facial_rating": result.get("facial_rating", "-"),
        "image_ratio_score": result.get("image_ratio_score", "-"),
        "proportion_notes": result.get("proportion_notes", []),
        "professional_summary": result.get("professional_summary", ""),
        "skin_information": result.get("skin_information", []),
        "routine": result.get("routine", []),
        "care_plan": result.get("care_plan", []),
        "notes": result.get("notes", ""),
    }


def analyze_with_ollama(image_base64, brands):
    base_url = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")

    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"

    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "gemma3"),
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

    with urllib.request.urlopen(request_obj, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data.get("message", {}).get("content", "")

    return parse_json_result(content)


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

    if not image or image.filename == "":
        return jsonify({"ok": False, "message": "Upload image first"}), 400

    if not allowed_file(image.filename):
        return jsonify({"ok": False, "message": "Use PNG JPG JPEG WEBP"}), 400

    image_bytes = image.read()

    try:
        result = analyze_with_ollama(
            image_bytes_to_base64(image_bytes),
            load_brands(),
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(
                    f"AI request failed: {exc}"
                ),
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
    )
