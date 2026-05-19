import re
import pathlib
from typing import List, Dict, Any, Tuple

HEADING_RE = re.compile(r'^(#{3,6})\s+(.*)$')
LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')
BULLET_RE = re.compile(r'^\*\s+(.*)$')
DEFAULT_VALUE_RE = re.compile(r'\*\*Default Value:\*\*\s*(.+)$')

# Normalizes a document name into a stable id fragment.
def normalize_id_name(name: str) -> str:
    return re.sub(r'[^a-z0-9_]+', '_', name.strip().lower())

# Splits a markdown file into section blocks.
def split_section_blocks(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_key = None
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            _, title = m.groups()
            key = title.strip().lower()
            current_key = key
            sections.setdefault(key, [])
        else:
            if current_key is not None:
                sections[current_key].append(line.rstrip('\n'))
    return sections

# Finds the item name and its section content.
def detect_name_and_sections(text: str):
    lines = text.splitlines()
    name = None
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m and m.group(1) == '###':
            name = m.group(2).strip()
            break
    if not name:
        return None, split_section_blocks(lines)
    sections = split_section_blocks(lines)
    return name, sections

# Parses an action document into structured fields.
def parse_action(name: str, sections: Dict[str, List[str]]) -> Dict[str, Any]:
    # Category
    category: List[str] = []
    for line in sections.get('category', []):
        line = line.strip()
        if line:
            category = [p.strip() for p in line.split(',')]
            break

    # Description
    desc_lines = [l.strip() for l in sections.get('description', []) if l.strip()]
    description_en = ' '.join(desc_lines)

    # Arguments table
    arg_lines = sections.get('argument', [])
    arguments: List[Dict[str, Any]] = []
    in_table = False
    for line in arg_lines:
        if '|' in line:
            cols = [c.strip() for c in line.split('|') if c.strip()]
            if cols and cols[0].lower() == 'name':
                in_table = True
                continue
            if in_table and len(cols) >= 4:
                arguments.append({
                    "name": cols[0],
                    "description_en": cols[1],
                    "type": cols[2],
                    "modifier": cols[3]
                })

    # Valid contexts
    valid_contexts: List[str] = []
    for line in sections.get('valid contexts', []):
        line = line.strip()
        if line:
            if ':' in line:
                line = line.split(':', 1)[1]
            valid_contexts = [p.strip().rstrip('.') for p in line.split('and')]
            break

    # Supported platforms/OS
    def parse_links(sec_key: str) -> List[str]:
        vals: List[str] = []
        for l in sections.get(sec_key, []):
            for m in LINK_RE.finditer(l):
                vals.append(m.group(1))
        return vals

    supported_platforms = parse_links('supported platforms')
    supported_os = parse_links('supported os')

    # Applicable settings
    applicable_setting_names: List[str] = []
    for l in sections.get('applicable built-in settings', []):
        for m in LINK_RE.finditer(l):
            applicable_setting_names.append(m.group(1))
    applicable_setting_ids = ['bis.' + normalize_id_name(n) for n in applicable_setting_names]

    # Applicable controls
    applicable_controls: List[str] = []
    for l in sections.get('applicable controls', []):
        l = l.strip()
        if l:
            applicable_controls = [p.strip() for p in l.split(',')]
            break

    # Sample code (first fenced block)
    code_examples = []
    in_code = False
    code_lines: List[str] = []
    for l in sections.get('sample code', []):
        if '```' in l:
            in_code = not in_code
            continue
        if in_code and l.strip():
            code_lines.append(l.strip())
    if code_lines:
        code_examples.append({
            "title": "default",
            "language": "plain",
            "lines": code_lines
        })

    # Notes
    notes_en: List[str] = []
    for l in sections.get('notes', []):
        m = BULLET_RE.match(l.strip())
        if m:
            notes_en.append(m.group(1).strip())

    return {
        "kind": "action",
        "name_en": name,
        "category": category,
        "reference": name,
        "description_en": description_en,
        "arguments": arguments,
        "return_value": "None",
        "valid_contexts": valid_contexts,
        "supported_platforms": supported_platforms,
        "supported_os": supported_os,
        "applicable_setting_ids": applicable_setting_ids,
        "applicable_controls": applicable_controls,
        "code_examples": code_examples,
        "notes_en": notes_en,
    }

# Parses a setting document into structured fields.
def parse_setting(name: str, sections: Dict[str, List[str]]) -> Dict[str, Any]:
    # Category
    category = None
    for line in sections.get('category', []):
        line = line.strip()
        if line:
            category = line
            break

    # Description
    desc_lines = [l.strip() for l in sections.get('description', []) if l.strip()]
    description_en = ' '.join(desc_lines)

    # Allowable values
    allowable_groups = []
    for l in sections.get('allowable values', []):
        m = BULLET_RE.match(l.strip())
        if m:
            txt = m.group(1)
            parts = [p.strip() for p in txt.split(',')]
            if len(parts) >= 2:
                allowable_groups.append(parts)

    # Default value
    default_value = None
    for l in sections.get('allowable values', []):
        m = DEFAULT_VALUE_RE.match(l.strip())
        if m:
            default_value = m.group(1).strip()
            break

    supported_platforms = []
    for l in sections.get('supported platforms', []):
        for m in LINK_RE.finditer(l):
            supported_platforms.append(m.group(1))

    supported_os = []
    for l in sections.get('supported os', []):
        for m in LINK_RE.finditer(l):
            supported_os.append(m.group(1))

    # Sample code
    code_examples = []
    in_code = False
    code_lines: List[str] = []
    for l in sections.get('sample code', []):
        if '```' in l:
            in_code = not in_code
            continue
        if in_code and l.strip():
            code_lines.append(l.strip())
    if code_lines:
        code_examples.append({
            "title": "default",
            "language": "plain",
            "lines": code_lines
        })

    notes_en: List[str] = []
    for l in sections.get('notes', []):
        m = BULLET_RE.match(l.strip())
        if m:
            notes_en.append(m.group(1).strip())

    return {
        "kind": "setting",
        "name_en": name,
        "category": category,
        "description_en": description_en,
        "allowable_values_groups": allowable_groups,
        "default_value": default_value,
        "supported_platforms": supported_platforms,
        "supported_os": supported_os,
        "code_examples": code_examples,
        "notes_en": notes_en,
    }

# Parses a markdown page according to its document kind.
def parse_markdown_with_kind(text: str, kind: str) -> Dict[str, Any]:
    name, sections = detect_name_and_sections(text)
    if not name:
        return {}
    if kind == "action":
        return parse_action(name, sections)
    return parse_setting(name, sections)

# Creates the base record for one KB item.
def initial_record(kind: str, slug: str, folder_category: str) -> Dict[str, Any]:
    prefix = "bia" if kind == "action" else "bis"
    return {
        "id": f"{prefix}.{slug.replace('-', '_')}",
        "kind": kind,
        "slug": slug,
        "folder_category": folder_category,
        "name_en": None,
        "name_ja": None,
        "description_en": None,
        "description_ja": None,
        "raw_en": None,
        "raw_ja": None,
    }

# Merges one locale version into a shared KB record.
def merge_locale(record: Dict[str, Any],
                 parsed: Dict[str, Any],
                 locale: str,
                 raw_text: str):
    suffix = "_en" if locale.upper() == "EN" else "_ja"
    record[f"name{suffix}"] = parsed.get("name_en")
    record[f"description{suffix}"] = parsed.get("description_en")
    record[f"notes{suffix}"] = parsed.get("notes_en")
    record[f"code_examples{suffix}"] = parsed.get("code_examples")
    record[f"raw{suffix}"] = raw_text

    for key in [
        "category", "reference", "return_value", "valid_contexts",
        "supported_platforms", "supported_os",
        "applicable_setting_ids", "applicable_controls",
        "allowable_values_groups", "default_value", "arguments"
    ]:
        if key in parsed and parsed[key] is not None and key not in record:
            record[key] = parsed[key]

# Ingests all ABT markdown pages under the source root.
def ingest_all(root: pathlib.Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for locale in ("EN", "JA"):
        for dir_name, kind in (("BIA", "action"), ("BIS", "setting")):
            base = root / locale / dir_name
            if not base.exists():
                continue
            for index_path in base.rglob("index.md"):
                rel = index_path.relative_to(base)
                parts = rel.parts
                if len(parts) < 2:
                    continue
                folder_category = parts[0]
                slug = parts[-2]
                key = (kind, slug)
                if key not in records:
                    records[key] = initial_record(kind, slug, folder_category)
                text = index_path.read_text(encoding="utf-8")
                parsed = parse_markdown_with_kind(text, kind)
                if not parsed:
                    continue
                merge_locale(records[key], parsed, locale, text)
    return records
