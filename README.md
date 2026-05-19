# BIA/BIS RAG Assistant

Technical spec and diagrams:

- [docs/hybrid-rag-technical-spec.md](docs/hybrid-rag-technical-spec.md)
- [docs/api-reference-ingest-rag.md](docs/api-reference-ingest-rag.md)

Support article ingestion:

- [support_urls.txt](support_urls.txt)

1. Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set env:

```bash
export OPENAI_API_KEY=sk-...
export DOCS_ROOT=/absolute/path/to/docs_root
```

Optional support-article settings:

```bash
export SUPPORT_URLS_FILE=/absolute/path/to/support_urls.txt
# or provide URLs directly, separated by newlines or commas
export SUPPORT_URLS="https://support.testarchitect.com/en/support/solutions/articles/4000022356-how-to-run-several-tests-one-after-another"
```

3. Build KB:

```bash
python -m ingest.build_kb
```

To ingest only the support articles without rebuilding the markdown KB:

```bash
export SUPPORT_ONLY_BUILD=1
python -m ingest.build_kb
```

### How support articles work

The app now treats TestArchitect support pages as a separate source type.

- ABT markdown still powers the BIA/BIS action and setting reference data.
- Support articles are ingested as summary-first records with the original article URL stored as the primary reference.
- The answer path keeps support articles concise and leads with the summary instead of forcing them into the action/setting template.
- This keeps how-to pages useful for command-heavy workflows without mixing them into the BIA/BIS canonical records.

4. Run the web chat UI:

```bash
python server.py
```

Then open:

```bash
http://127.0.0.1:8000
```

If you want to share it on your local network, run the server on a machine others can reach and use that machine's IP address with port `8000`.

5. Or run CLI:

```bash
python cli_chat.py
```

### Recommended workflow

1. Keep your ABT markdown under `ABT/`.
2. Add support article URLs to `support_urls.txt`.
3. Run `python -m ingest.build_kb` to rebuild the KB.
4. Use the web UI or CLI to ask either:
   - BIA/BIS action and setting questions
   - support how-to questions that should return a short summary and the article reference link
