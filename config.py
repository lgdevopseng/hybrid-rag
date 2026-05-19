import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# where your EN/JA markdown lives
DOCS_ROOT = Path(os.getenv("DOCS_ROOT", BASE_DIR / "ABT"))

# optional support-article URL list for how-to articles
SUPPORT_URLS_FILE = Path(os.getenv("SUPPORT_URLS_FILE", str(BASE_DIR / "support_urls.txt")))
SUPPORT_URLS = os.getenv("SUPPORT_URLS", "")
SUPPORT_ONLY_BUILD = os.getenv("SUPPORT_ONLY_BUILD", "").strip().lower() in {"1", "true", "yes", "on"}

# sqlite db path
DB_PATH = BASE_DIR / "abt_kb.sqlite"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-4.1-mini")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
