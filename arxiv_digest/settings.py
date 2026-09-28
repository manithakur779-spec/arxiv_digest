import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
VECTOR_STORE_PATH = PROJECT_ROOT / ".paper_digest_chroma"
MODEL = "qwen/qwen3.8-27b"

load_dotenv(dotenv_path=ENV_FILE, override=False)


def create_llm_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(f"Set GROQ_API_KEY in {ENV_FILE} to a Groq API key.")
    return Groq(api_key=api_key)