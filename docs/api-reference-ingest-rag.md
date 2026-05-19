# Ingest and RAG API Reference

This document covers the functions in `ingest/` and `rag/` that make up the KB build, retrieval, and answer generation flow.

## 1. `ingest.parser`

| Function | Purpose |
| --- | --- |
| `normalize_id_name(name)` | Normalize a display name into a stable slug-like identifier fragment. |
| `split_section_blocks(lines)` | Split markdown text into section buckets keyed by heading text. |
| `detect_name_and_sections(text)` | Extract the item name from the first `###` heading and return the parsed sections. |
| `parse_action(name, sections)` | Convert an action markdown document into a structured action record. |
| `parse_setting(name, sections)` | Convert a setting markdown document into a structured setting record. |
| `parse_markdown_with_kind(text, kind)` | Parse a markdown file as either an action or setting record. |
| `initial_record(kind, slug, folder_category)` | Create the base record used while merging EN and JA locales. |
| `merge_locale(record, parsed, locale, raw_text)` | Merge one locale's parsed fields into the shared record. |
| `ingest_all(root)` | Walk the ABT tree, parse EN/JA docs, and return normalized records keyed by logical item. |

### `ingest.parser` notes

- `parse_action()` captures action metadata such as arguments, valid contexts, supported platforms/OS, related settings, controls, notes, and sample code.
- `parse_setting()` captures setting metadata such as allowable values, default value, supported platforms/OS, notes, and sample code.
- `ingest_all()` skips files without a valid `###` heading.

## 2. `ingest.support_articles`

| Function | Purpose |
| --- | --- |
| `load_support_urls(path)` | Load support-article URLs from a plain text file. |
| `extract_article_meta(url)` | Derive the article id, slug, and locale from a support URL. |
| `strip_tags_fragment(html_text)` | Reduce HTML to readable text lines. |
| `extract_title(html_text, fallback)` | Pull the browser title or fall back to a slug-derived title. |
| `normalize_lines(text)` | Clean and filter extracted support-article lines. |
| `summarize_support_lines(lines)` | Build a short summary from the article body. |
| `fetch_support_article(url)` | Fetch one support article and normalize it into a KB record. |
| `ingest_support_articles(urls)` | Convert a list of support URLs into KB records. |

### `ingest.support_articles` notes

- Support articles are stored as summary-first records with a canonical source URL.
- The ingester is designed to be additive so support URLs can be added without reworking the ABT markdown parser.

## 3. `ingest.build_kb`

| Function | Purpose |
| --- | --- |
| `get_embedding(text)` | Request an embedding vector for KB search from the configured OpenAI model. |
| `ensure_schema(conn)` | Create the SQLite tables and FTS index if they do not exist. |
| `build_kb()` | Rebuild the SQLite KB from the markdown docs, embeddings, and relations. |

### `ingest.build_kb` notes

- `build_kb()` writes to `items`, `items_fts`, `embeddings`, and `relations`.
- Action-to-setting relations are created from `applicable_setting_ids`.
- If `support_urls.txt` or `SUPPORT_URLS` is provided, `build_kb()` also ingests summary-first support articles with their canonical URLs.
- Set `SUPPORT_ONLY_BUILD=1` to ingest only the support articles without walking the ABT markdown tree.

## 4. `rag.db`

| Function | Purpose |
| --- | --- |
| `get_conn()` | Open a SQLite connection to the KB database with row access by name. |
| `fetch_items(ids)` | Load stored item JSON for one or more item ids. |
| `fetch_relations_from(src_ids)` | Load graph relations for one or more source item ids. |

## 5. `rag.retrieval`

| Function | Purpose |
| --- | --- |
| `embed_query(query)` | Generate an embedding vector for a user query. |
| `cosine_sim(a, b)` | Compute cosine similarity between two vectors. |
| `build_fts_query(query)` | Turn a free-form query into a safe FTS5 query string. |
| `build_focus_query(query)` | Strip common stopwords to produce a tighter focus query. |
| `find_exact_match_ids(conn, query)` | Find exact item matches by name or slug. |
| `hybrid_search(query, k=5)` | Rank candidates using embeddings, FTS5, and exact-match boosting. |
| `expand_with_relations(seed_ids, max_extra=2)` | Add a small number of related items from graph edges. |
| `retrieve_for_query(query, k=5, use_graph=True)` | Return the final retrieved item records for a question. |

### `rag.retrieval` notes

- `hybrid_search()` uses embeddings and FTS5 together.
- `retrieve_for_query()` marks exact matches and optionally expands through graph relations.

## 6. `rag.chat`

| Function | Purpose |
| --- | --- |
| `build_history_context(history)` | Compress prior turns into a short conversation summary. |
| `is_followup_query(query)` | Detect whether a query looks like a follow-up. |
| `infer_focus_from_history(history)` | Pull the most recent item focus from prior assistant turns. |
| `detect_response_language(query)` | Pick English or Japanese response language from the query. |
| `normalize_text(value)` | Normalize text for matching and comparison. |
| `unique_preserve_order(values)` | Deduplicate items while keeping the original order. |
| `humanize_identifier(value)` | Convert internal ids into display text. |
| `extract_reference_terms(text)` | Extract search terms used for reference ranking. |
| `item_search_text(item)` | Build a searchable text blob from an item record. |
| `gather_related_reference_items(items)` | Pull directly related setting records from action records. |
| `lookup_exact_item_by_name(name)` | Fetch a single item by exact name, slug, or display form. |
| `item_is_named_in_text(item, text)` | Check whether an item is explicitly named in text. |
| `canonical_item_key(item)` | Build the canonical key used to merge duplicate logical items. |
| `merge_duplicate_items(items)` | Merge duplicate logical items and combine useful fields. |
| `build_source_path(item, target_language)` | Build the local markdown source path for an item. |
| `build_official_doc_url(item, target_language)` | Build the official TestArchitect docs URL for an item. |
| `build_reference_entries(items, target_language, query="", answer="")` | Select and rank the reference links shown to the user. |
| `extract_sample_code_lines(raw_text)` | Extract sample code lines from raw markdown. |
| `extract_sample_code_sections(raw_text)` | Extract sample-code sections as grouped blocks. |
| `split_sample_sections(sample_lines)` | Split extracted sample lines into blocks by blank line. |
| `split_sample_row(line)` | Split one sample row into table cells. |
| `format_sample_code_tables(sample_sections)` | Render a sample section as markdown table rows. |
| `strip_model_example_usage(answer)` | Remove any example-usage block the model tried to invent. |
| `get_item_display_name(item)` | Return the best display name for an item. |
| `select_primary_item_for_example(items, query, answer="")` | Choose the main item whose example should be shown. |
| `select_relevant_sample_section(item, sections, query="")` | Choose the best sample block inside an item. |
| `build_example_usage_section(item, target_language, query="")` | Render the final localized example-usage section. |
| `append_example_usage(answer, items, target_language, query="")` | Append the deterministic example section to the answer. |
| `build_context_snippets(items, target_language="en")` | Compress retrieved items into the prompt context. |
| `answer_query_with_meta(query, history=None)` | Run retrieval + generation and return answer plus metadata. |
| `answer_query(query, history=None)` | Convenience wrapper that returns only the final answer text. |

### `rag.chat` notes

- `build_reference_entries()` is intentionally conservative and prefers explicit, relevant references.
- `append_example_usage()` now appends a single example from the primary item rather than mixing graph-related examples.
- Support articles are treated as summary-first sources with a direct canonical reference link.
- `answer_query_with_meta()` returns the answer, clarifying state, review flag, references, and merged items.

## 7. Suggested entry points

- `python -m ingest.build_kb`
- `python cli_chat.py`
- `python server.py`
