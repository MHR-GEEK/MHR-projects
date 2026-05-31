import os

from waitress import serve

from app import app, ensure_data_file


if __name__ == "__main__":
    ensure_data_file()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8081"))
    serve(app, host=host, port=port, threads=8)
