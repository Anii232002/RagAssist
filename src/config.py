import os
from pathlib import Path

OPEN_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

PERSIST_DIRECTORY = os.getenv(
     "PERSIST_DIRECTORY",
    str(Path(__file__).resolve().parent.parent / "storage" / "chroma_store")
)

# Default data root
DATA_ROOT = os.getenv(
    "DATA_ROOT",
    str(Path(__file__).resolve().parent.parent / "data")
)