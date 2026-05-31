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
    """Ensure the data directory and brands file exist."""
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
    """Load approved brands from the brands.json file."""
    ensure_data_file()
    return json.loads(BRANDS_FILE.read_text(encoding="utf-8"))


def save_brands(brands):
    """Save the list of approved brands to the brands.json file."""
    ensure_data_file()
    normalized = sorted({brand.strip() for brand in brands if brand.strip()})
    BRANDS_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_data_url(file_storage):
    """Convert a file storage object to a data URL."""
    mime_type = file_storage.mimetype or "image/png"
    image_bytes = file_storage.read()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_data_url(image_bytes, mime_type):
    """Convert image bytes to a data URL."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_base64(image_bytes):
    """Convert image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("ascii")


def fallback_result(message):
    """Return a fallback result when AI is not available."""
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
        "notes": message,
    }


def build_analysis_prompt(brands):
    """Build the prompt for AI analysis based on approved brands."""
    brand_text = ", ".join(brands) if brands else "No brands have been approved yet"
    return f"""
You are an educational skincare assistant, not a medical diagnostic tool.
Analyze the uploaded face image only for visible, non-sensitive skincare observations.
Do not identify the person, infer protected traits, diagnose disease, or promise results.
Only recommend products from this admin-approved brand list: {brand_text}.

Return strict JSON with this schema:
{{
  "skin_type": "one of: oily, dry, combination, normal, unclear",
  "concerns": ["short visible skincare concerns, or unclear"],
  "sensitivity": "low, moderate, high, or unclear",
  "psl_score": "subjective PSL-style visual harmony score from 1.0 to 10.0, or unclear",
  "facial_rating": "brief subjective facial aesthetic rating such as low, average, above average, high, or unclear",
  "image_ratio_score": "visible facial proportion/photo ratio score from 1 to 100, or unclear",
  "proportion_notes": ["brief notes about visible symmetry, facial thirds, lighting, angle, and photo quality"],
  "routine": [
    {{"step": "AM cleanser", "recommendation": "brand-safe educational recommendation"}},
    {{"step": "AM moisturizer", "recommendation": "brand-safe educational recommendation"}},
    {{"step": "AM sunscreen", "recommendation": "brand-safe educational recommendation"}},
    {{"step": "PM cleanser", "recommendation": "brand-safe educational recommendation"}},
    {{"step": "PM treatment", "recommendation": "brand-safe educational recommendation"}},
    {{"step": "PM moisturizer", "recommendation": "brand-safe educational recommendation"}}
  ],
  "notes": "brief educational disclaimer and when to see a dermatologist"
}}
"""


def parse_json_result(raw_text):
    """Parse the raw JSON result from the AI service."""
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
    """Normalize the result to a consistent format."""
    return {
        "ai_available": ai_available,
        "skin_type": result.get("skin_type", "-"),
        "concerns": result.get("concerns") if isinstance(result.get("concerns"), list) else [],
        "sensitivity": result.get("sensitivity", "-"),
        "psl_score": result.get("psl_score", "-"),
        "facial_rating": result.get("facial_rating", "-"),
        "image_ratio_score": result.get("image_ratio_score", "-"),
        "proportion_notes": result.get("proportion_notes")
        if isinstance(result.get("proportion_notes"), list)
        else [],
        "routine": result.get("routine") if isinstance(result.get("routine"), list) else [],
        "notes": result.get("notes", ""),
    }


def analyze_with_openai(image_data_url, brands):
    """Analyze the image using OpenAI."""
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
    """Analyze the image using Ollama."""
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
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI service returned HTTP {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the AI service. Check your connection and restart the app."
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"AI service returned no message content: {data}")
    return parse_json_result(content)


def chat_with_ollama(message):
    """Send a message to the Ollama chat model and return the response."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"
    model = os.environ.get("OLLAMA_CHAT_MODEL", os.environ.get("OLLAMA_MODEL", "gemma3:12b"))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a very brief AI skincare assistant. Reply in 1 to 2 short sentences. "
                    "Avoid long lists unless the user asks. Give educational skincare guidance, "
                    "product routine help, and app usage help. Do not diagnose medical conditions. "
                    "Tell users to see a dermatologist for urgent, painful, spreading, bleeding, "
                    "infected, or persistent symptoms."
                ),
            },
            {"role": "user", "content": message},
        ],
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_obj = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI service returned HTTP {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach the AI service.") from exc

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"AI service returned no chat content: {data}")
    return content


def configured_provider():
    """Determine the configured AI provider."""
    provider = os.environ.get("AI_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return ""


def ai_status_label():
    """Return the AI status label based on the configured provider."""
    provider = configured_provider()
    if provider == "ollama":
        return "AI ready"
    if provider == "openai":
        return "AI ready"
    return "AI key missing"


@app.get("/")
def index():
    """Render the main index page."""
    return render_template(
        "index.html",
        brands=load_brands(),
        ai_configured=bool(configured_provider()),
        ai_status=ai_status_label(),
    )


@app.post("/admin/login")
def admin_login():
    """Admin login endpoint."""
    password = request.json.get("password", "")
    expected = os.environ.get("ADMIN_PASSWORD", "admin123")
    if password != expected:
        return jsonify({"ok": False, "message": "Wrong admin password"}), 401
    session["admin"] = True
    return jsonify({"ok": True, "message": "Logged in"})


@app.post("/admin/brands")
def add_brand():
    """Add a new brand to the approved list."""
    if not session.get("admin"):
        return jsonify({"ok": False, "message": "Admin login required"}), 403
    brand = request.json.get("brand", "").strip()
    if not brand:
        return jsonify({"ok": False, "message": "Brand name is required"}), 4
    brands = load_brands()
    brands.append(brand)
    save_brands(brands)
    return jsonify({"ok": True, "brands": load_brands()})


@app.delete("/admin/brands/")
def delete_brand():
    """Delete a brand from the approved list."""
    if not session.get("admin"):
        return jsonify({"ok": False, "message": "Admin login required"}), 403
    brand = request.args.get("brand", "").strip()
    if not brand:
        return jsonify({"ok": False, "message": "Brand name is required"}), 400
    brands = [item for item in load_brands() if item.lower() != brand.lower()]
    save_brands(brands)
    return jsonify({"ok": True, "brands": load_brands()})


@app.post("/analyze")
def analyze():
    """Analyze the uploaded image using the configured AI provider."""
    image = request.files.get("image")
    if not image or image.filename == "":
        return jsonify({"ok": False, "message": "Upload a face image first"}), 400
    if not allowed_file(image.filename):
        return jsonify({"ok": False, "message": "Use PNG, JPG, JPEG, or WEBP"}), 400

    image_bytes = image.read()
    provider = configured_provider()
    if not provider:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(
                    "AI is not configured. Set AI_PROVIDER and the matching API key, restart the app, and analyze again."
                ),
            }
        )

    try:
        if provider == "ollama":
            result = analyze_with_ollama(image_bytes_to_base64(image_bytes), load_brands())
        elif provider == "openai":
            mime_type = image.mimetype or "image/png"
            result = analyze_with_openai(image_bytes_to_data_url(image_bytes, mime_type), load_brands())
        else:
            return jsonify(
                {
                    "ok": True,
                    "result": fallback_result(f"Unsupported AI_PROVIDER: {provider}"),
                }
            )
    except Exception as exc:
        return jsonify(
            {
                "ok": True,
                "result": fallback_result(f"AI request failed: {exc}"),
            }
        )

    return jsonify({"ok": True, "result": result})


@app.post("/chat")
def chat():
    """Chat with the AI using the configured provider."""
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify({"ok": False, "message": "Type a message first"}), 400

    provider = configured_provider()
    if provider != "ollama":
        return jsonify(
            {
                "ok": False,
                "message": "Chat is not configured.",
            }
        ), 400

    try:
        reply = chat_with_ollama(message)
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Chat failed: {exc}"}), 500

    return jsonify({"ok": True, "reply": reply})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Dermatologist Assistant")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ensure_data_file()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=args.debug)
