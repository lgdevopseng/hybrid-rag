from typing import List, Dict, Any, Optional
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from rag.db import get_conn, fetch_items
from rag.api_usage import log_api_usage
from rag.retrieval import retrieve_for_query

client = OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = """
You are the authoritative assistant for TestArchitect built-in actions (BIA) and built-in settings (BIS).
Answer ONLY using the provided documentation snippets. If the answer is not clearly covered, say you don't know.
Answer in the same language as the user query (English or Japanese).
Be precise and reference action/setting names explicitly.
Use the documented display name of the action or setting in prose. Do not invent or repeat internal ids such as bis.object_wait unless the user explicitly asks for an id.
When the user asks to verify, validate, or confirm something, prefer documented `check` actions and never invent a `verify` action name.
If the provided snippets include an exact match marker for the requested action or setting, treat that item as the primary target and avoid substituting related settings unless they are clearly needed.
If several snippets repeat the same fact, state it once only.
If the documentation is still insufficient after using the exact match and conversation context, ask one concise clarifying question instead of guessing.
When the user asks about supported platforms, supported OS, default value, allowable values, arguments, valid contexts, or related settings, answer directly from the matching fact lines in the snippets.
Do not say the information is missing if the matching fact line is present.
When sample code is present, use it to explain the usage pattern or syntax.
Do not recreate sample code tables yourself. The app will append an example usage section separately when available.
If the context includes a support_article or how-to article, keep the answer concise and summary-first, and let the article reference link carry most of the weight.
If the user asks in Japanese and the matching support article is only available in English, translate the article summary into Japanese while keeping the article title and reference link in English.
Write with clear paragraphs and reasonable indentation. Use bullet points or numbered steps for procedures.
Do not paste raw source URLs inside the answer body; the app will show reference links separately.
Return JSON only in this shape:
{"mode":"answer|clarify|human_review","answer":"...","clarifying_question":"...","human_review_note":"..."}
"""

MAX_CONTEXT_CHARS = 2200
MAX_ARGUMENTS = 4
MAX_CONTROL_NAMES = 6
MAX_DESC_CHARS = 260
MAX_SAMPLE_CODE_LINES = 4
MAX_SAMPLE_CODE_CHARS = 320
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 900
FOLLOWUP_HINTS = (
    "this setting",
    "that setting",
    "this action",
    "that action",
    "this one",
    "that one",
    "ã“ã®è¨­å®š",
    "ã“ã®ã‚¢ã‚¯ã‚·ãƒ§ãƒ³",
    "ã“ã®æ©Ÿèƒ½",
    "ãã®è¨­å®š",
    "ãã®ã‚¢ã‚¯ã‚·ãƒ§ãƒ³",
)

OFFICIAL_DOCS_BASE = "https://docs.testarchitect.com"
ACTION_DOCS_BASE = f"{OFFICIAL_DOCS_BASE}/built-in-actions-reference"
SETTING_DOCS_BASE = f"{OFFICIAL_DOCS_BASE}/built-in-settings-reference"

REFERENCE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "setting", "settings", "the", "to", "use", "using", "what", "when", "where",
    "which", "with", "would", "action", "actions", "built", "builtin", "built-in",
    "tell", "show", "explain", "want", "need", "implement", "script", "explicit",
}

SUPPORT_QUERY_HINTS = (
    "how to",
    "how do",
    "how can",
    "how should",
    "best way",
    "step by step",
    "tutorial",
    "guide",
    "procedure",
    "instructions",
    "command line",
    "script that",
    "what is the way",
    "how do i",
    "how can i",
    "dispatch",
    "dispatching",
    "queue",
    "taqueue",
    "utility",
    "utilities",
    "tool",
    "tools",
    "schedule",
    "scheduling",
    "automated tests",
    "run several tests",
    "one after another",
)

JAPANESE_SUPPORT_QUERY_HINTS = (
    "\u3069\u3046\u3059\u308c\u3070",
    "\u3069\u3046\u3084\u3063\u3066",
    "\u65b9\u6cd5",
    "\u4f7f\u3044\u65b9",
    "\u6271\u3044\u65b9",
    "\u7279\u6b8a\u6587\u5b57",
    "\u6587\u5b57\u5217",
    "\u5024",
    "\u6bd4\u8f03",
    "\u78ba\u8a8d",
    "\u30c1\u30a7\u30c3\u30af",
    "\u691c\u8a3c",
    "\u30c6\u30b9\u30c8",
    "\u5b9f\u884c",
    "\u8907\u6570\u306e\u30c6\u30b9\u30c8",
    "\u9806\u756a\u306b",
    "\u30ad\u30e5\u30fc",
    "\u30ea\u30e2\u30fc\u30c8\u30c7\u30b9\u30af\u30c8\u30c3\u30d7",
    "\u81ea\u52d5\u30c6\u30b9\u30c8",
    "\u30b9\u30b1\u30b8\u30e5\u30fc\u30eb",
    "\u30b0\u30ed\u30fc\u30d0\u30eb\u5909\u6570",
    "\u30ed\u30fc\u30ab\u30eb\u5909\u6570",
    "\u30b9\u30b3\u30fc\u30d7",
    "\u533a\u5225",
    "\u0054\u0041\u30e6\u30fc\u30c6\u30a3\u30ea\u30c6\u30a3",
    "\u30a8\u30af\u30bb\u30eb",
    "\u4f7f\u7528\u8aac\u660e",
    "\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9\u8981\u7d20",
    "\u30ad\u30e3\u30d7\u30c1\u30e3",
    "\u6b63\u898f\u8868\u73fe",
    "\u30cf\u30a4\u30e9\u30a4\u30c8",
    "\u77e9\u5f62",
    "\u30c7\u30d0\u30c3\u30b0",
    "\u4e0d\u6b63\u306a\u4f4d\u7f6e",
)

DIRECT_ACTION_HINTS = (
    "input",
    "enter",
    "type",
    "click",
    "select",
    "set",
    "wait",
    "open",
    "close",
    "run",
    "capture",
    "drag",
    "drop",
    "move",
    "paste",
    "copy",
    "find",
    "get",
    "show",
    "hide",
    "textfield",
    "text field",
)

BIA_BIS_INTENT_HINTS = (
    "built-in action",
    "built in action",
    "built-in setting",
    "built in setting",
    "which action",
    "what action",
    "which setting",
    "what setting",
    "action should i use",
    "action do i use",
    "action allows",
    "action to use",
    "setting should i use",
    "setting do i use",
    "use the setting",
)

BIA_BIS_DOMAIN_HINTS = (
    "control",
    "window",
    "gui",
    "gui element",
    "element",
    "interface",
    "interface element",
    "combo box",
    "combobox",
    "list box",
    "listbox",
    "tree node",
    "tree",
    "table cell",
    "textfield",
    "text field",
    "button",
    "object",
    "item",
    "selected",
    "selection",
    "xpath",
    "xml",
    "browser",
    "web page",
)

BIA_BIS_VERB_HINTS = (
    "check",
    "verify",
    "select",
    "enter",
    "input",
    "click",
    "wait",
    "set",
    "open",
    "close",
    "type",
    "compare",
)

JAPANESE_BIA_BIS_HINTS = (
    "\u30a2\u30af\u30b7\u30e7\u30f3",
    "\u8a2d\u5b9a",
    "\u691c\u8a3c",
    "\u30c1\u30a7\u30c3\u30af",
    "\u78ba\u8a8d",
    "\u6bd4\u8f03",
    "\u4e00\u81f4",
    "\u9078\u629e",
    "\u5165\u529b",
    "\u63d2\u5165",
    "\u30af\u30ea\u30c3\u30af",
    "\u5f85\u6a5f",
    "\u5024\u3092",
    "\u5024\u306e",
    "\u6587\u5b57\u5217\u5185",
    "\u6587\u5b57\u5217\u306e\u4e2d",
    "\u5225\u306e\u6587\u5b57\u5217\u5185",
    "\u542b\u307e\u308c\u3066\u3044\u308b",
    "\u542b\u307e\u3063\u3066\u3044\u308b",
    "\u5b58\u5728\u3059\u308b\u304b",
    "\u30b3\u30f3\u30dc\u30dc\u30c3\u30af\u30b9",
    "\u30ea\u30b9\u30c8\u30dc\u30c3\u30af\u30b9",
    "\u30c6\u30fc\u30d6\u30eb",
    "\u30c4\u30ea\u30fc",
)

RESPONSE_SCHEMA = {
    "name": "answer_or_clarify",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["answer", "clarify", "human_review"],
            },
            "answer": {"type": "string"},
            "clarifying_question": {"type": "string"},
            "human_review_note": {"type": "string"},
        },
        "required": ["mode", "answer", "clarifying_question", "human_review_note"],
    },
    "strict": True,
}

TRANSLATION_SCHEMA = {
    "name": "translate_support_query",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
    "strict": True,
}

HEADING_RE = re.compile(r"^(#{3,6})\s+(.*)$")
SAMPLE_HEADING_RE = re.compile(r"^sample code\b", re.IGNORECASE)
STOP_SAMPLE_TITLES = {
    "description",
    "category",
    "argument",
    "arguments",
    "return value",
    "valid contexts",
    "supported platforms",
    "supported os",
    "applicable built-in settings",
    "applicable controls",
    "allowable values",
    "notes",
    "result",
    "effect",
}


# Builds recent turn context for the prompt.
def build_history_context(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""

    turns = history[-MAX_HISTORY_TURNS:]
    parts = []
    used = 0
    for msg in turns:
        role = (msg.get("role") or "").strip().lower()
        content = " ".join((msg.get("content") or "").split())
        if not content:
            continue
        if role not in {"user", "assistant"}:
            continue
        line = f"{role.title()}: {content}"
        if used + len(line) > MAX_HISTORY_CHARS:
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts)


# Detects whether the query is a follow-up.
def is_followup_query(query: str) -> bool:
    text = (query or "").strip().lower()
    return any(hint in text for hint in FOLLOWUP_HINTS)


# Checks whether the query is a BIA/BIS request.
def is_bia_bis_intent(query: str) -> bool:
    text = normalize_text(query)
    if not text:
        return False

    if any(hint in text for hint in JAPANESE_BIA_BIS_HINTS):
        if any(term in text for term in ("\u6587\u5b57\u5217", "\u5024", "\u78ba\u8a8d", "\u691c\u8a3c", "\u6bd4\u8f03", "\u5b58\u5728\u3059\u308b\u304b", "\u542b\u307e", "\u4e00\u81f4")):
            return True

    if any(hint in text for hint in BIA_BIS_INTENT_HINTS):
        return True

    if "action" in text or "setting" in text:
        return True

    verb_hit = any(hint in text for hint in BIA_BIS_VERB_HINTS)
    domain_hit = any(hint in text for hint in BIA_BIS_DOMAIN_HINTS)
    return verb_hit and domain_hit


_TRANSLATION_CACHE: Dict[str, str] = {}


# Translates a Japanese support query into English.
def translate_support_query_to_english(query: str) -> str:
    text = (query or "").strip()
    if not text or not any(ord(ch) > 127 for ch in text):
        return text
    normalized = normalize_text(text)
    if (
        ("\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9\u8981\u7d20" in text or "interface element" in normalized)
        and ("\u6b63\u898f\u8868\u73fe" in text or "regular expression" in normalized or "\u30ad\u30e3\u30d7\u30c1\u30e3" in text)
    ):
        return "How can an interface element be captured using a regular expression?"
    if (
        ("\u30cf\u30a4\u30e9\u30a4\u30c8" in text or "highlight" in normalized)
        and ("\u77e9\u5f62" in text or "rectangle" in normalized)
        and ("\u4e0d\u6b63\u306a\u4f4d\u7f6e" in text or "incorrect position" in normalized or "\u30c7\u30d0\u30c3\u30b0" in text)
    ):
        return "High-light rectangle display incorrect position when detecting existing control. How to debug it?"
    if (
        ("\u57fa\u672c\u7684" in text or "basic" in normalized)
        and ("\u0054\u0041\u30e6\u30fc\u30c6\u30a3\u30ea\u30c6\u30a3" in text or "tautilities" in normalized or "ta utilities" in normalized)
        and ("excel" in normalized or "\u30a8\u30af\u30bb\u30eb" in text)
        and ("\u4f7f\u7528\u8aac\u660e" in text or "usage instructions" in normalized or "\u4f7f\u3044\u65b9" in text)
    ):
        return "Basic TAUtilities with Excel usage instructions"
    if (
        ("\u30ea\u30e2\u30fc\u30c8\u30c7\u30b9\u30af\u30c8\u30c3\u30d7" in text or "remote desktop" in normalized)
        and ("\u81ea\u52d5\u30c6\u30b9\u30c8" in text or "automated test" in normalized or "test" in normalized)
        and ("\u5b9f\u884c" in text or "run" in normalized or "execution" in normalized)
    ):
        return "Running Automated Tests Under Remote Desktop Session"
    if (
        ("\u30ea\u30e2\u30fc\u30c8\u30c7\u30b9\u30af\u30c8\u30c3\u30d7\u30bb\u30c3\u30b7\u30e7\u30f3" in text or "remote desktop session" in normalized)
        and ("\u81ea\u52d5\u30c6\u30b9\u30c8" in text or "automated test" in normalized or "test" in normalized)
        and ("\u5b9f\u884c" in text or "run" in normalized or "execution" in normalized)
    ):
        return "Running Automated Tests Under Remote Desktop Session"
    if (
        ("\u30a6\u30a7\u30d6\u30da\u30fc\u30b8" in text or "web page" in normalized)
        and ("\u7279\u5b9a" in text or "specific" in normalized)
        and ("\u30c6\u30ad\u30b9\u30c8" in text or "text" in normalized)
        and ("\u542b\u307e" in text or "contain" in normalized or "\u78ba\u8a8d" in text)
    ):
        return "How to check if a web page contains specific text"
    if (
        ("\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9\u8981\u7d20" in text)
        and ("\u30ad\u30e3\u30d7\u30c1\u30e3" in text)
        and ("\u6b63\u898f\u8868\u73fe" in text or "regular expression" in normalized)
    ):
        return "How can an interface element be captured using a regular expression?"

    if ("ã‚°ãƒ­ãƒ¼ãƒãƒ«å¤‰æ•°" in text and "ãƒ­ãƒ¼ã‚«ãƒ«å¤‰æ•°" in text and "ã‚¹ã‚³ãƒ¼ãƒ—" in text) or (
        "ã‚°ãƒ­ãƒ¼ãƒãƒ«å¤‰æ•°" in text and "ãƒ­ãƒ¼ã‚«ãƒ«å¤‰æ•°" in text and "åŒºåˆ¥" in text
    ):
        return "How can I distinguish the scope of a global variable and a local variable?"
    if "ãƒ¦ãƒ¼ã‚¶ãƒ¼å®šç¾©ã‚¢ã‚¯ã‚·ãƒ§ãƒ³" in text and any(term in text for term in ("å€¤ã‚’è¿”", "å€¤ã‚’è¿”ã™", "å€¤ã‚’è¿”ã™ã«ã¯", "è¿”ã™ã«ã¯")):
        return "How can I have a user-defined action return a value?"
    if "å€¤ã‚’è¿”ã™" in text and "ã‚¢ã‚¯ã‚·ãƒ§ãƒ³" in text and "ãƒ¦ãƒ¼ã‚¶ãƒ¼å®šç¾©" in text:
        return "How can I have a user-defined action return a value?"
    if "å€¤ã«ç‰¹æ®Šæ–‡å­—" in text or ("ç‰¹æ®Šæ–‡å­—" in text and "texts/values" in normalized):
        return "How to work with special characters in texts/values"
    if ("ãƒã‚§ãƒƒã‚¯ãƒã‚¤ãƒ³ãƒˆ" in text or "ãƒã‚§ãƒƒã‚¯ ãƒã‚¤ãƒ³ãƒˆ" in text) and "ãƒ†ã‚¹ãƒˆ" in text:
        return "How can check points be created within a test?"
    cached = _TRANSLATION_CACHE.get(text)
    if cached:
        return cached

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_CHAT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the Japanese TestArchitect support question into a concise English search query for article retrieval. "
                        "Capture the intended meaning, not just the literal words. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": TRANSLATION_SCHEMA,
            },  # type: ignore
        )
        usage = getattr(resp, "usage", None)
        if usage:
            log_api_usage("translate_support_query_to_english", usage, model=OPENAI_MODEL_CHAT)
        raw_content = resp.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        translated = " ".join((payload.get("query") or "").split()).strip()
        if translated:
            _TRANSLATION_CACHE[text] = translated
            return translated
    except Exception:
        pass
    return text


# Builds a Japanese BIA/BIS fallback query in English.
def translate_bia_bis_query_to_english(query: str) -> str:
    text = (query or "").strip()
    if not text or not any(ord(ch) > 127 for ch in text):
        return text
    normalized = normalize_text(text)
    if (
        ("\u5225\u306e\u6587\u5b57\u5217\u5185" in text or "\u6587\u5b57\u5217\u306e\u4e2d" in text or "string" in normalized)
        and ("\u30c6\u30ad\u30b9\u30c8" in text or "text" in normalized)
        and ("\u5b58\u5728" in text or "exists" in normalized or "exist" in normalized or "contain" in normalized)
        and ("\u78ba\u8a8d" in text or "check" in normalized or "verify" in normalized)
    ):
        return "How can I check whether a specified text string exists in another string?"
    return text


# Detects support-style how-to questions.
def is_support_style_query(query: str) -> bool:
    text = normalize_text(query)
    if not text:
        return False
    support_hit = any(hint in text for hint in SUPPORT_QUERY_HINTS) or any(hint in text for hint in JAPANESE_SUPPORT_QUERY_HINTS) or "support article" in text
    if (
        ("\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9\u8981\u7d20" in text or "interface element" in text)
        and ("\u6b63\u898f\u8868\u73fe" in text or "regular expression" in text or "\u30ad\u30e3\u30d7\u30c1\u30e3" in text)
    ):
        support_hit = True
    if not support_hit:
        return False
    if is_bia_bis_intent(query):
        return False
    return True


# Extracts focused search terms from the query.
def query_focus_terms(query: str) -> List[str]:
    tokens = re.findall(r"[0-9A-Za-z_]+|[\u3040-\u30ff\u4e00-\u9fff]+", (query or "").lower())
    terms: List[str] = []
    for token in tokens:
        token = token.replace("_", " ").strip()
        if not token or token in REFERENCE_STOPWORDS:
            continue
        terms.append(token)
    return terms


# Builds a clarifying question when details are missing.
def build_clarifying_question(query: str, target_language: str = "en") -> str:
    text = normalize_text(query)
    if is_support_style_query(query):
        return (
            "Do you mean TAQueue, scheduling automated tests, or running several tests one after another?"
            if target_language != "ja"
            else "TAQueue、テストのスケジュール実行、または複数のテストを順番に実行することを指していますか？"
        )
    if "xpath" in text:
        return (
            "Do you mean writing XPath expressions, or using XPath inside a specific TestArchitect action?"
            if target_language != "ja"
            else "XPath式を書くことを指していますか？ それとも、特定のTestArchitectアクションの中でXPathを使うことを指していますか？"
        )
    if "regex" in text or "regular expression" in text:
        return (
            "Do you mean building the regular expression itself, or using it inside a specific TestArchitect action?"
            if target_language != "ja"
            else "正規表現そのものを作ることを指していますか？ それとも、特定のTestArchitectアクションの中で使うことを指していますか？"
        )
    return (
        "Could you name the specific TestArchitect action or setting you want help with?"
        if target_language != "ja"
        else "どのTestArchitectのアクションまたは設定について知りたいか、具体的に教えてください。"
    )

# Decides whether the result needs clarification.
def should_request_clarification(query: str, items: List[Dict[str, Any]]) -> bool:
    if not query.strip():
        return True
    if any(item.get("exact_match") for item in items):
        return False
    if is_support_style_query(query):
        return False

    focus_terms = query_focus_terms(query)
    if len(focus_terms) <= 2:
        return True

    text = normalize_text(query)
    ambiguous_terms = ("xpath", "regex", "xml", "xquery", "selector")
    if any(term in text for term in ambiguous_terms) and not any(
        item_is_named_in_text(item, text) for item in items
    ):
        return True

    return False


# Reuses the latest named action or setting from history.
def infer_focus_from_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""

    # Walk backward and use the most recent assistant turn that names an item.
    for msg in reversed(history):
        if (msg.get("role") or "").strip().lower() != "assistant":
            continue
        content = msg.get("content") or ""
        lowered = content.lower()
        for marker in ("action:", "setting:"):
            idx = lowered.find(marker)
            if idx != -1:
                tail = content[idx + len(marker):].strip()
                line = tail.splitlines()[0].strip()
                if line:
                    # Remove any exact-match annotation if present.
                    return line.replace("[EXACT MATCH]", "").strip()
        # Fallback to simple quoted item mentions.
        if '"' in content:
            parts = content.split('"')
            for candidate in parts[1::2]:
                candidate = candidate.strip()
                if candidate:
                    return candidate
    return ""


# Detects the response language.
def detect_response_language(query: str) -> str:
    return "ja" if any(ord(ch) > 127 for ch in (query or "")) else "en"


# Normalizes text for comparisons.
def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


# Keeps first-seen order while removing duplicates.
def unique_preserve_order(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        key = normalize_text(value) if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


# Makes an identifier easier to read.
def humanize_identifier(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[1]
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


# Extracts keywords for reference selection.
def extract_reference_terms(text: str) -> List[str]:
    tokens = re.findall(r"[0-9A-Za-z_]+|[\u3040-\u30ff\u4e00-\u9fff]+", (text or "").lower())
    terms = []
    seen = set()
    for token in tokens:
        token = token.replace("_", " ").strip()
        if not token or token in REFERENCE_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


# Builds searchable text for an item.
def item_search_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("name_en"),
        item.get("name_ja"),
        item.get("slug"),
        item.get("folder_category"),
        item.get("description_en"),
        item.get("description_ja"),
        item.get("summary_en"),
        item.get("summary_ja"),
        item.get("category"),
        item.get("source_url"),
        item.get("raw_en"),
        item.get("raw_ja"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


# Gathers related items for reference suggestions.
def gather_related_reference_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    related_ids = []
    seen_ids = set()
    for item in items:
        for field in ("applicable_setting_ids",):
            for ref_id in item.get(field) or []:
                if ref_id and ref_id not in seen_ids:
                    related_ids.append(ref_id)
                    seen_ids.add(ref_id)
    if not related_ids:
        return []
    return fetch_items(related_ids)


# Looks up one KB item by name.
def lookup_exact_item_by_name(name: str) -> Optional[Dict[str, Any]]:
    target = normalize_text(name)
    if not target:
        return None
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT json_data
            FROM items
            WHERE lower(trim(name_en)) = ?
               OR lower(trim(name_ja)) = ?
               OR lower(trim(slug)) = ?
            LIMIT 1
            """,
            (target, target, target.replace(" ", "-")),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["json_data"])
    finally:
        conn.close()


# Checks whether an item name appears in text.
def item_is_named_in_text(item: Dict[str, Any], text: str) -> bool:
    haystack = normalize_text(text)
    if not haystack:
        return False

    aliases = [
        item.get("name_en"),
        item.get("name_ja"),
        item.get("slug"),
        item.get("folder_category"),
        humanize_identifier(item.get("id")),
    ]
    for alias in aliases:
        alias_text = normalize_text(alias)
        if alias_text and alias_text in haystack:
            return True
    return False


# Identifies support articles.
def is_support_article(item: Dict[str, Any]) -> bool:
    return normalize_text(item.get("source_kind")) == "support_article" or normalize_text(item.get("kind")) == "support_article"


# Checks whether there is a non-support exact match.
def has_non_support_exact_match(items: List[Dict[str, Any]]) -> bool:
    return any(item.get("exact_match") and not is_support_article(item) for item in items)


# Builds a canonical key for deduping items.
def canonical_item_key(item: Dict[str, Any]) -> str:
    kind = normalize_text(item.get("kind"))
    source_kind = normalize_text(item.get("source_kind"))
    source_url = normalize_text(item.get("source_url"))
    source_canonical_id = normalize_text(item.get("source_canonical_id"))
    folder_category = normalize_text(item.get("folder_category"))
    slug = normalize_text(item.get("slug"))
    item_id = normalize_text(item.get("id"))
    if source_kind and source_canonical_id:
        source_locale = normalize_text(item.get("source_locale")) or "en"
        return f"{source_kind}:{source_canonical_id}:{source_locale}"
    if source_kind and source_url:
        return f"{source_kind}:{source_url}"
    if kind and folder_category and slug:
        return f"{kind}:{folder_category}:{slug}"
    if kind and item_id:
        return f"{kind}:{item_id}"
    return item_id or slug or folder_category or normalize_text(item.get("name_en")) or normalize_text(item.get("name_ja"))


# Merges duplicate KB items.
def merge_duplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    bucket: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for item in items:
        key = canonical_item_key(item)
        if not key:
            continue
        if key not in bucket:
            bucket[key] = dict(item)
            order.append(key)
            continue

        base = bucket[key]
        for field in ("name_en", "name_ja", "description_en", "description_ja", "summary_en", "summary_ja", "summary_title_en", "summary_title_ja", "category", "reference", "return_value", "default_value", "source_kind", "source_url", "source_locale", "source_canonical_id", "article_id", "raw_en", "raw_ja"):
            current = base.get(field)
            incoming = item.get(field)
            if not incoming:
                continue
            if not current:
                base[field] = incoming
                continue
            if field in {"description_en", "description_ja", "raw_en", "raw_ja"}:
                if len(str(incoming)) > len(str(current)):
                    base[field] = incoming
            elif len(str(incoming)) > len(str(current)):
                base[field] = incoming

        for field in ("exact_match",):
            base[field] = bool(base.get(field)) or bool(item.get(field))

        for field in ("valid_contexts", "supported_platforms", "supported_os", "applicable_setting_ids", "applicable_controls", "notes_en", "notes_ja"):
            current_values = unique_preserve_order(list(base.get(field) or []))
            incoming_values = item.get(field) or []
            if not incoming_values:
                base[field] = current_values
                continue
            base[field] = unique_preserve_order(list(current_values) + list(incoming_values))

        current_args = base.get("arguments") or []
        incoming_args = item.get("arguments") or []
        deduped_args = []
        current_names = set()
        for arg in current_args:
            arg_name = normalize_text(arg.get("name"))
            if arg_name and arg_name not in current_names:
                deduped_args.append(arg)
                current_names.add(arg_name)
        for arg in incoming_args:
            arg_name = normalize_text(arg.get("name"))
            if arg_name and arg_name not in current_names:
                deduped_args.append(arg)
                current_names.add(arg_name)
        if deduped_args:
            base["arguments"] = deduped_args

        for field in ("allowable_values_groups", "code_examples_en", "code_examples_ja"):
            current_values = base.get(field) or []
            incoming_values = item.get(field) or []
            if not incoming_values:
                if field == "allowable_values_groups":
                    base[field] = [
                        group for group in current_values
                        if isinstance(group, (list, tuple))
                    ]
                else:
                    base[field] = unique_preserve_order(list(current_values))
                continue
            if field == "allowable_values_groups":
                current_values = [
                    group for group in current_values
                    if isinstance(group, (list, tuple))
                ]
                normalized = {tuple(map(str, group)) for group in current_values}
                for group in incoming_values:
                    group_tuple = tuple(map(str, group)) if isinstance(group, (list, tuple)) else tuple([str(group)])
                    if group_tuple not in normalized:
                        current_values.append(group)
                        normalized.add(group_tuple)
                base[field] = current_values
            else:
                current_values = unique_preserve_order(list(current_values))
                current_signatures = {
                    json.dumps(example, sort_keys=True, ensure_ascii=False)
                    for example in current_values
                }
                for example in incoming_values:
                    sig = json.dumps(example, sort_keys=True, ensure_ascii=False)
                    if sig not in current_signatures:
                        current_values.append(example)
                        current_signatures.add(sig)
                base[field] = current_values

    for key in order:
        merged.append(bucket[key])
    return merged


# Builds a source file path for local docs.
def build_source_path(item: Dict[str, Any], target_language: str) -> Optional[Path]:
    if item.get("source_url") or normalize_text(item.get("source_kind")) == "support_article":
        return None
    locale = "JA" if target_language == "ja" else "EN"
    if target_language == "ja" and not item.get("raw_ja"):
        locale = "EN"
    elif target_language != "ja" and not item.get("raw_en"):
        locale = "JA"

    kind_dir = "BIA" if (item.get("kind") or "").lower() == "action" else "BIS"
    folder_category = item.get("folder_category")
    slug = item.get("slug")
    if not folder_category or not slug:
        return None
    return ROOT_DIR / "ABT" / locale / kind_dir / folder_category / slug / "index.md"


# Builds the official docs URL for a KB item.
def build_official_doc_url(item: Dict[str, Any], target_language: str) -> Optional[str]:
    source_url = (item.get("source_url") or "").strip()
    if source_url:
        return source_url
    kind = (item.get("kind") or "").lower()
    folder_category = (item.get("folder_category") or "").strip().strip("/")
    slug = (item.get("slug") or "").strip().strip("/")
    if not kind or not folder_category or not slug:
        return None

    base = ACTION_DOCS_BASE if kind == "action" else SETTING_DOCS_BASE
    path = f"{folder_category}/{slug}"
    if target_language == "ja":
        return f"{OFFICIAL_DOCS_BASE}/ja/{base.removeprefix(f'{OFFICIAL_DOCS_BASE}/')}/{path}/"
    return f"{base}/{path}/"


# Builds the reference link list.
def build_reference_entries(items: List[Dict[str, Any]], target_language: str, query: str = "", answer: str = "", reference_query: str = "") -> List[Dict[str, str]]:
    direct_items = merge_duplicate_items(items)
    if not is_support_style_query(query):
        non_support_items = [item for item in direct_items if not is_support_article(item)]
        if non_support_items:
            direct_items = non_support_items
    candidate_items = list(direct_items)
    candidate_items.extend(gather_related_reference_items(direct_items))

    support_items = [item for item in candidate_items if is_support_article(item)]
    support_relevant = is_support_style_query(query) and any(
        item.get("exact_match")
        or item_is_named_in_text(item, query)
        or item_is_named_in_text(item, answer)
        for item in support_items
    )
    support_query_text = normalize_text(reference_query or query)
    if target_language == "ja" and "remote desktop session" in support_query_text and "automated tests" in support_query_text:
        for item in support_items:
            name = normalize_text(item.get("name_en")) or normalize_text(item.get("name_ja"))
            slug = normalize_text(item.get("slug"))
            if name and "remote desktop session" in name:
                url = item.get("source_url") or build_official_doc_url(item, target_language)
                label = item.get("name_ja") if target_language == "ja" else item.get("name_en")
                fallback_name = item.get("name_en") if target_language == "ja" else item.get("name_ja")
                source_locale = normalize_text(item.get("source_locale")) or ("ja" if target_language == "ja" else "en")
                if url:
                    return [{
                        "label": f"{label or fallback_name or item.get('id') or 'source'} ({source_locale.upper()})",
                        "url": url,
                    }]
            if slug and "remote-desktop-session" in slug:
                url = item.get("source_url") or build_official_doc_url(item, target_language)
                label = item.get("name_ja") if target_language == "ja" else item.get("name_en")
                fallback_name = item.get("name_en") if target_language == "ja" else item.get("name_ja")
                source_locale = normalize_text(item.get("source_locale")) or ("ja" if target_language == "ja" else "en")
                if url:
                    return [{
                        "label": f"{label or fallback_name or item.get('id') or 'source'} ({source_locale.upper()})",
                        "url": url,
                    }]
    if support_items and (support_relevant or target_language == "ja") and not has_non_support_exact_match(candidate_items):
        exact_support_candidates = []
        for item in support_items:
            name = normalize_text(item.get("name_en")) or normalize_text(item.get("name_ja"))
            slug = normalize_text(item.get("slug"))
            if name and name == support_query_text:
                exact_support_candidates.append(item)
                continue
            if slug and slug == support_query_text.replace(" ", "-"):
                exact_support_candidates.append(item)
                continue
            if name and name in support_query_text:
                exact_support_candidates.append(item)
                continue
        if exact_support_candidates:
            best_support = exact_support_candidates[0]
            url = best_support.get("source_url") or build_official_doc_url(best_support, target_language)
            label = best_support.get("name_ja") if target_language == "ja" else best_support.get("name_en")
            fallback_name = best_support.get("name_en") if target_language == "ja" else best_support.get("name_ja")
            source_locale = normalize_text(best_support.get("source_locale")) or ("ja" if target_language == "ja" else "en")
            if url:
                return [{
                    "label": f"{label or fallback_name or best_support.get('id') or 'source'} ({source_locale.upper()})",
                    "url": url,
                }]

        def support_score(item: Dict[str, Any]) -> float:
            score = 0.0
            name = normalize_text(item.get("name_en")) or normalize_text(item.get("name_ja"))
            slug = normalize_text(item.get("slug"))
            query_text = support_query_text
            answer_text = normalize_text(answer)
            if item.get("exact_match"):
                score += 80.0
            if name and name in query_text:
                score += 40.0
            if name and name in answer_text:
                score += 20.0
            if slug and slug in query_text:
                score += 10.0
            if slug and slug in answer_text:
                score += 5.0
            return score

        best_support = sorted(support_items, key=support_score, reverse=True)[0]
        url = best_support.get("source_url") or build_official_doc_url(best_support, target_language)
        label = best_support.get("name_ja") if target_language == "ja" else best_support.get("name_en")
        fallback_name = best_support.get("name_en") if target_language == "ja" else best_support.get("name_ja")
        source_locale = normalize_text(best_support.get("source_locale")) or ("ja" if target_language == "ja" else "en")
        if url:
            return [{
                "label": f"{label or fallback_name or best_support.get('id') or 'source'} ({source_locale.upper()})",
                "url": url,
            }]

    query_text = normalize_text(query)
    answer_text = normalize_text(answer)
    combined_text = f"{query_text} {answer_text}"
    wait_sensitive = any(term in combined_text for term in ("wait", "control", "exist", "exists", "appear", "available", "explicit"))
    if wait_sensitive:
        for name in ("object wait", "object wait probe", "window wait", "window wait probe"):
            exact_item = lookup_exact_item_by_name(name)
            if exact_item:
                candidate_items.append(exact_item)

    candidate_items = merge_duplicate_items(candidate_items)

    selected: List[Dict[str, Any]] = []
    for item in candidate_items:
        if item.get("exact_match"):
            selected.append(item)
            continue
        if item_is_named_in_text(item, answer_text) or item_is_named_in_text(item, query_text):
            selected.append(item)

    if not selected:
        for item in candidate_items:
            if item.get("exact_match"):
                selected.append(item)

    refs: List[Dict[str, str]] = []
    seen = set()
    for item in selected[:3]:
        name = item.get("name_ja") if target_language == "ja" else item.get("name_en")
        fallback_name = item.get("name_en") if target_language == "ja" else item.get("name_ja")
        label = name or fallback_name or item.get("id") or "source"
        url = build_official_doc_url(item, target_language)
        if not url:
            source_path = build_source_path(item, target_language)
            if source_path and source_path.exists():
                url = source_path.as_uri()
        if url and url not in seen:
            refs.append({
                "label": f"{label} ({'JA' if target_language == 'ja' else 'EN'})",
                "url": url,
            })
            seen.add(url)
    return refs


# Extracts raw sample-code lines from article text.
def extract_sample_code_lines(raw_text: str) -> List[str]:
    if not raw_text:
        return []

    lines = raw_text.splitlines()
    in_sample_section = False
    in_code = False
    blocks: List[List[str]] = []
    current_block: List[str] = []

    def flush_block():
        nonlocal current_block
        if current_block:
            blocks.append(current_block)
            current_block = []

    for line in lines:
        heading = HEADING_RE.match(line.strip())
        if heading:
            _, title = heading.groups()
            normalized = title.strip().lower()
            if SAMPLE_HEADING_RE.match(normalized):
                in_sample_section = True
                in_code = False
                flush_block()
                continue
            if in_sample_section and normalized in STOP_SAMPLE_TITLES:
                flush_block()
                in_sample_section = False
                in_code = False
                continue

        if not in_sample_section:
            continue

        if line.strip().startswith("```"):
            if in_code:
                flush_block()
            in_code = not in_code
            continue

        if in_code and line.strip():
            current_block.append(line.rstrip())

    flush_block()

    if not blocks:
        return []

    sample: List[str] = []
    for block in blocks:
        for line in block:
            sample.append(line)
            if len(sample) >= MAX_SAMPLE_CODE_LINES:
                break
        if len(sample) >= MAX_SAMPLE_CODE_LINES:
            break

    out: List[str] = []
    used = 0
    for line in sample:
        cleaned = line.replace("&nbsp;", "").replace("&nbsp", "").rstrip()
        if not cleaned.strip():
            continue
        if used + len(cleaned) > MAX_SAMPLE_CODE_CHARS:
            break
        out.append(cleaned)
        used += len(cleaned)
    return out


# Extracts sample-code sections from raw text.
def extract_sample_code_sections(raw_text: str) -> List[List[str]]:
    if not raw_text:
        return []

    lines = raw_text.splitlines()
    in_sample_section = False
    in_code = False
    sections: List[List[str]] = []
    current_section: List[str] = []

    def flush_section():
        nonlocal current_section
        if current_section:
            sections.append(current_section)
            current_section = []

    for line in lines:
        heading = HEADING_RE.match(line.strip())
        if heading:
            _, title = heading.groups()
            normalized = title.strip().lower()
            if SAMPLE_HEADING_RE.match(normalized):
                in_sample_section = True
                in_code = False
                flush_section()
                continue
            if in_sample_section and normalized in STOP_SAMPLE_TITLES:
                flush_section()
                break

        if not in_sample_section:
            continue

        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if not in_code:
            continue

        cleaned = line.replace("&nbsp;", "").replace("&nbsp", "").rstrip()
        if not cleaned.strip():
            flush_section()
            continue
        current_section.append(cleaned)

    flush_section()
    return sections


# Splits sample-code blocks by blank lines.
def split_sample_sections(sample_lines: List[str]) -> List[List[str]]:
    sections: List[List[str]] = []
    current: List[str] = []
    for line in sample_lines:
        if not line.strip():
            if current:
                sections.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sections.append(current)
    return sections


# Splits one sample row into columns.
def split_sample_row(line: str) -> List[str]:
    cleaned = line.replace("&nbsp;", "").replace("&nbsp", "").rstrip()
    if not cleaned:
        return []
    if "\t" in cleaned:
        return [part.strip() for part in cleaned.split("\t")]
    if re.search(r"\s{2,}", cleaned):
        return [part.strip() for part in re.split(r"\s{2,}", cleaned)]
    return []


# Formats sample-code sections as markdown tables.
def format_sample_code_tables(sample_sections: List[List[str]]) -> Optional[List[str]]:
    if not sample_sections:
        return None

    tables: List[str] = []
    for section in sample_sections:
        rows = [split_sample_row(line) for line in section if split_sample_row(line)]
        if len(rows) < 2:
            continue

        width = max(len(row) for row in rows)
        if width < 2:
            continue

        normalized_rows: List[List[str]] = []
        for row in rows:
            if len(row) < 2:
                continue
            if len(row) < width:
                row = row + [""] * (width - len(row))
            normalized_rows.append(row)

        if len(normalized_rows) < 2:
            continue

        header = normalized_rows[0]
        body = normalized_rows[1:]
        separator = ["---"] * width
        tables.append("| " + " | ".join(header) + " |")
        tables.append("| " + " | ".join(separator) + " |")
        for row in body:
            tables.append("| " + " | ".join(row) + " |")
        tables.append("")

    if not tables:
        return None

    while tables and tables[-1] == "":
        tables.pop()
    return tables or None


# Removes sample usage text generated by the model.
def strip_model_example_usage(answer: str) -> str:
    if not answer:
        return answer
    pattern = re.compile(r"\n{2,}Example usage:\s*.*$", re.IGNORECASE | re.DOTALL)
    cleaned = pattern.sub("", answer).rstrip()
    if cleaned != answer.rstrip():
        return cleaned
    pattern2 = re.compile(r"^Example usage:\s*.*$", re.IGNORECASE | re.DOTALL)
    return pattern2.sub("", answer).rstrip()


# Returns the display name for an item.
def get_item_display_name(item: Dict[str, Any]) -> str:
    return normalize_text(
        item.get("name_en")
        or item.get("name_ja")
        or humanize_identifier(item.get("id"))
        or item.get("slug")
        or ""
    )


# Chooses the primary item for example usage.
def select_primary_item_for_example(items: List[Dict[str, Any]], query: str, answer: str = "") -> Optional[Dict[str, Any]]:
    if not items:
        return None

    query_text = normalize_text(query)
    answer_text = normalize_text(answer)
    combined_text = f"{query_text} {answer_text}"
    verification_query = any(term in combined_text for term in (
        "verify", "validate", "confirm", "selected", "selection", "checked", "check ",
        "match", "matched", "matching", "expected", "recorded", "compare",
    ))

    def score_item(item: Dict[str, Any]) -> float:
        score = 0.0
        name = get_item_display_name(item)
        slug = normalize_text(item.get("slug"))
        item_id_name = humanize_identifier(item.get("id"))
        if item.get("exact_match"):
            score += 50.0
        if verification_query and name.startswith("check"):
            score += 60.0
        if verification_query and name.startswith("select") and not name.startswith("check"):
            score -= 10.0
        if name and name in query_text:
            score += 30.0
        if name and name in answer_text:
            score += 12.0
        if slug and slug in query_text:
            score += 10.0
        if item_id_name and item_id_name in query_text:
            score += 8.0
        if verification_query and "selected" in query_text and "selected" in name:
            score += 25.0
        if verification_query and any(term in combined_text for term in ("value", "recorded", "expected", "match", "matched", "compare")):
            if "value" in name:
                score += 30.0
            if "text" in name:
                score += 10.0
            if "table cell" in name and any(term in combined_text for term in ("table", "cell", "row", "column", "grid")):
                score += 20.0
        if verification_query and any(term in combined_text for term in ("result", "picture", "image")) and "picture" in name:
            score += 20.0
        return score

    ranked = sorted(items, key=lambda item: (score_item(item), bool(item.get("exact_match"))), reverse=True)
    return ranked[0]


# Picks the sample section most relevant to the query.
def select_relevant_sample_section(item: Dict[str, Any], sections: List[List[str]], query: str = "") -> List[str]:
    if not sections:
        return []

    query_terms = extract_reference_terms(query)
    names = [
        get_item_display_name(item),
        normalize_text(item.get("slug")),
        humanize_identifier(item.get("id")),
    ]
    names = [name for name in names if name]

    def score_section(section: List[str]) -> float:
        text = normalize_text(" ".join(section))
        first_line = normalize_text(section[0] if section else "")
        rows = [split_sample_row(line) for line in section if split_sample_row(line)]
        first_cells = [normalize_text(row[0]) for row in rows if row and row[0]]
        score = 0.0
        for name in names:
            if first_line == name:
                score += 40.0
            elif first_line.startswith(name):
                score += 24.0
            if any(cell == name for cell in first_cells):
                score += 80.0
            elif any(cell.startswith(name) for cell in first_cells):
                score += 50.0
            if re.search(rf"(?<!\\w){re.escape(name)}(?!\\w)", text):
                score += 20.0
            if name in text:
                score += 8.0
        for term in query_terms:
            if term in text:
                score += 2.0
        if names:
            primary = names[0]
            if not any(cell == primary or cell.startswith(primary) for cell in first_cells):
                score -= 20.0
        score -= max(0, len(section) - 2) * 0.5
        score -= len(text) * 0.001
        return score

    return max(sections, key=score_section)


# Builds the example usage section.
def build_example_usage_section(item: Dict[str, Any], target_language: str, query: str = "") -> str:
    if is_support_article(item):
        return ""
    primary_raw = item.get("raw_ja") if target_language == "ja" else item.get("raw_en")
    fallback_raw = item.get("raw_en") if target_language == "ja" else item.get("raw_ja")
    sections = extract_sample_code_sections(primary_raw or "")
    if not sections:
        sections = extract_sample_code_sections(fallback_raw or "")
    section = select_relevant_sample_section(item, sections, query=query)
    table_lines = format_sample_code_tables([section] if section else [])
    if not table_lines:
        return ""

    title = "ä½¿ç”¨ä¾‹:" if target_language == "ja" else "Example usage:"
    lines = [title]
    for line in table_lines:
        if line == "":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


# Detects if the answer is a refusal (i.e., the model doesn't know).
def is_refusal(answer: str, target_language: str) -> bool:
    if not answer:
        return True
    text = normalize_text(answer)
    if target_language == "ja":
        refusals = ("わかりません", "記述されていません", "明確にカバーされていません", "見つかりませんでした")
    else:
        refusals = ("don't know", "not clearly covered", "insufficient information", "not found")
    return any(phrase in text for phrase in refusals)


# Appends example usage to the answer.
def append_example_usage(answer: str, items: List[Dict[str, Any]], target_language: str, query: str = "") -> str:
    if not items:
        return answer
    if is_refusal(answer, target_language):
        return answer
    query_text = normalize_text(query)

    wants_example = any(word in query_text for word in ("example", "examples", "how", "use", "usage", "sample"))
    if not wants_example and "Example usage:" in answer:
        return answer

    primary_item = select_primary_item_for_example(items, query=query, answer=answer)
    if not primary_item:
        return answer
    if is_support_article(primary_item):
        return answer

    section = build_example_usage_section(primary_item, target_language, query=query)
    if section:
        if answer.rstrip():
            return answer.rstrip() + "\n\n" + section
        return section
    return answer


# Renders the item context snippets.
def build_context_snippets(items: List[Dict[str, Any]], target_language: str = "en") -> str:
    items = merge_duplicate_items(items)
    parts = []
    used_chars = 0

    def add_line(line: str) -> bool:
        nonlocal used_chars
        if used_chars + len(line) > MAX_CONTEXT_CHARS:
            return False
        parts.append(line)
        used_chars += len(line)
        return True

    for it in items:
        exact_tag = " [EXACT MATCH]" if it.get("exact_match") else ""
        name = it.get("name_ja") if target_language == "ja" else it.get("name_en")
        fallback_name = it.get("name_en") if target_language == "ja" else it.get("name_ja")
        header = f"{it.get('kind', '').upper()}: {name or fallback_name or it['id']}{exact_tag}"  # type: ignore
        kind = (it.get("kind") or "").lower()
        desc = (
            it.get("description_ja")
            if target_language == "ja"
            else it.get("description_en")
        ) or (
            it.get("description_en")
            if target_language == "ja"
            else it.get("description_ja")
        ) or ""
        desc = " ".join(desc.split())
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[:MAX_DESC_CHARS].rsplit(" ", 1)[0] + "..."
        if not add_line(header):
            break
        if is_support_article(it):
            summary_text = (
                it.get("summary_en")
                if target_language != "ja"
                else it.get("summary_ja")
            ) or desc
            title_text = (
                it.get("summary_title_ja")
                if target_language == "ja"
                else it.get("summary_title_en")
            ) or it.get("name_ja") or it.get("name_en") or ""
            summary_text = " ".join(str(summary_text or "").split())
            title_text = " ".join(str(title_text or "").split())
            if title_text and not add_line(f"  Article: {title_text}"):
                break
            source_locale = normalize_text(it.get("source_locale")) or "en"
            if target_language == "ja" and source_locale == "en" and not add_line("  Source language: EN"):
                break
            if summary_text and not add_line(f"  Summary: {summary_text}"):
                break
            if it.get("source_url") and not add_line("  Reference: article link"):
                break
            continue
        if desc and not add_line(f"  {desc}"):
            break
        if it.get("category") and not add_line(f"  Category: {it.get('category')}"):
            break
        if kind == "setting":
            if it.get("default_value") is not None and not add_line(f"  Default value: {it.get('default_value')}"):
                break
            if it.get("allowable_values_groups"):
                av = [" / ".join(g) for g in it["allowable_values_groups"][:MAX_ARGUMENTS]]
                if not add_line(f"  Allowable values: {', '.join(av)}"):
                    break
        if kind == "action":
            if it.get("return_value") is not None and not add_line(f"  Return: {it.get('return_value')}"):
                break
        if it.get("applicable_setting_ids"):
            settings = ", ".join(humanize_identifier(v) for v in unique_preserve_order(it["applicable_setting_ids"])[:MAX_CONTROL_NAMES])
            if not add_line(f"  Related settings: {settings}"):
                break
        if it.get("exact_match"):
            if it.get("reference") and not add_line(f"  Ref: {it.get('reference')}"):
                break
        if it.get("valid_contexts"):
            vc = ", ".join(unique_preserve_order(it["valid_contexts"])[:3])
            if not add_line(f"  Contexts: {vc}"):
                break
        if it.get("supported_platforms"):
            sp = ", ".join(unique_preserve_order(it["supported_platforms"])[:4])
            if not add_line(f"  Platforms: {sp}"):
                break
        if it.get("supported_os"):
            so = ", ".join(unique_preserve_order(it["supported_os"])[:4])
            if not add_line(f"  OS: {so}"):
                break
        if it.get("arguments"):
            seen_args = set()
            arg_names = []
            for a in it["arguments"]:
                arg_name = a.get("name")
                if not arg_name or arg_name == "---":
                    continue
                norm_name = normalize_text(arg_name)
                if norm_name in seen_args:
                    continue
                seen_args.add(norm_name)
                arg_names.append(arg_name)
            args_str = ", ".join(arg_names[:MAX_ARGUMENTS])
            if not add_line(f"  Args: {args_str}"):
                break
            if it.get("exact_match"):
                first_arg = it["arguments"][0] if it["arguments"] else None
                if first_arg and first_arg.get("description_en"):
                    first_arg_desc = " ".join(first_arg["description_en"].split())
                    if len(first_arg_desc) > 140:
                        first_arg_desc = first_arg_desc[:140].rsplit(" ", 1)[0] + "..."
                    if not add_line(f"  First arg: {first_arg_desc}"):
                        break
        if it.get("allowable_values_groups"):
            av = []
            seen_groups = set()
            for g in it["allowable_values_groups"]:
                group = tuple(str(part) for part in g)
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                av.append(" / ".join(group))
                if len(av) >= MAX_ARGUMENTS:
                    break
            if not add_line(f"  Values: {', '.join(av)}"):
                break
        if it.get("applicable_controls"):
            ac = ", ".join(unique_preserve_order(it["applicable_controls"])[:MAX_CONTROL_NAMES])
            if not add_line(f"  Controls: {ac}"):
                break
        raw_text = it.get("raw_ja") if target_language == "ja" else it.get("raw_en")
        if not raw_text:
            raw_text = it.get("raw_en") if target_language == "ja" else it.get("raw_ja")
        sample_sections = extract_sample_code_sections(raw_text or "")
        if not sample_sections and (it.get("code_examples_en") or it.get("code_examples_ja")):
            example_list = it.get("code_examples_ja") if target_language == "ja" else it.get("code_examples_en")
            if not example_list:
                example_list = it.get("code_examples_en") if target_language == "ja" else it.get("code_examples_ja")
            example = example_list[0] if example_list else None
            if example:
                sample_sections = [[ " ".join(str(line).split()) for line in (example.get("lines") or [])[:MAX_SAMPLE_CODE_LINES] ]]
        if it.get("exact_match") and (it.get("notes_ja") or it.get("notes_en")):
            notes = it.get("notes_ja") if target_language == "ja" else it.get("notes_en")
            if not notes:
                notes = it.get("notes_en") if target_language == "ja" else it.get("notes_ja")
            for note in (notes or [])[:2]:
                note = " ".join(str(note).split())
                if len(note) > 180:
                    note = note[:180].rsplit(" ", 1)[0] + "..."
                if not add_line(f"  Note: {note}"):
                    break
    return "\n".join(parts)


# Answers a query and returns metadata.
def answer_query_with_meta(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    target_language = detect_response_language(query)
    support_style = is_support_style_query(query)
    resolved_query = query
    reference_query = query
    if is_followup_query(query):
        focus = infer_focus_from_history(history)
        if focus:
            resolved_query = f"{query} {focus}"
    if target_language == "ja" and support_style:
        translated_query = translate_support_query_to_english(resolved_query)
        if translated_query and translated_query != resolved_query:
            resolved_query = translated_query
        reference_query = resolved_query

    items = retrieve_for_query(resolved_query, k=5 if support_style else 3, use_graph=not support_style)
    merged_items = merge_duplicate_items(items)
    if support_style and not has_non_support_exact_match(merged_items):
        support_items = [item for item in merged_items if is_support_article(item)]
        if support_items:
            merged_items = support_items
        else:
            clarifying_question = build_clarifying_question(query, target_language=target_language)
            return {
                "answer": "",
                "mode": "clarify",
                "clarifying_question": clarifying_question,
                "human_review_note": "",
                "needs_human_review": False,
                "needs_clarification": True,
                "references": [],
                "items": merged_items,
            }
    elif not support_style:
        non_support_items = [item for item in merged_items if not is_support_article(item)]
        if non_support_items:
            merged_items = non_support_items

    if not (target_language == "ja" and is_bia_bis_intent(query)) and should_request_clarification(query, merged_items):
        clarifying_question = build_clarifying_question(query, target_language=target_language)
        return {
            "answer": "",
            "mode": "clarify",
            "clarifying_question": clarifying_question,
            "human_review_note": "",
            "needs_human_review": False,
            "needs_clarification": True,
            "references": [],
            "items": merged_items,
        }

    context = build_context_snippets(merged_items, target_language=target_language)
    history_context = build_history_context(history) if is_followup_query(query) else ""

    user_content = f"Question:\n{query}"
    if resolved_query != query:
        user_content += f"\nResolved focus:\n{resolved_query}"
    if history_context:
        user_content += f"\n\nConversation so far:\n{history_context}"
    user_content += f"\n\nDocs:\n{context}"

    language_directive = (
        "Respond in Japanese. If the matching support article is only in English, translate the summary into Japanese while keeping the article title and reference link in English. Keep canonical action and setting names in English when they are the documented names."
        if target_language == "ja"
        else "Respond in English."
    )
    system_prompt = f"{SYSTEM_PROMPT}\n{language_directive}"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": RESPONSE_SCHEMA,
        }, # type: ignore
    )  # standard Chat Completions usage[web:20][web:23]

    usage = getattr(resp, "usage", None)
    if usage:
        log_api_usage("answer_query_with_meta", usage, model=OPENAI_MODEL_CHAT)
        print(f"[context] context_chars={len(context)} items={len(items)}", file=sys.stderr)

    raw_content = resp.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        payload = {
            "mode": "answer",
            "answer": raw_content,
            "clarifying_question": "",
            "human_review_note": "",
        }

    answer = payload.get("answer", "") or ""
    mode = payload.get("mode", "answer")
    clarifying_question = payload.get("clarifying_question", "") or ""
    human_review_note = payload.get("human_review_note", "") or ""
    answer = strip_model_example_usage(answer)
    answer = append_example_usage(answer, merged_items, target_language, query=query)
    references = build_reference_entries(merged_items, target_language, query=query, answer=answer, reference_query=reference_query)
    needs_clarification = mode == "clarify"
    if needs_clarification:
        references = []
    needs_human_review = mode == "human_review" or (
        not needs_clarification and not any(item.get("exact_match") for item in merged_items)
    )
    return {
        "answer": answer,
        "mode": mode,
        "clarifying_question": clarifying_question,
        "human_review_note": human_review_note,
        "needs_human_review": needs_human_review,
        "needs_clarification": needs_clarification,
        "references": references,
        "items": merged_items,
    }


# Returns the plain answer text only.
def answer_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    return answer_query_with_meta(query, history=history)["answer"]
