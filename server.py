import os
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rag.chat import answer_query_with_meta

app = FastAPI(title="BIA/BIS RAG Assistant")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BIA/BIS RAG Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d9dee8;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --user: #dbeafe;
      --assistant: #eef2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      max-width: 980px;
      margin: 0 auto;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .messages {
      flex: 1;
      min-height: 60vh;
      padding: 20px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 80%;
      padding: 12px 14px;
      border-radius: 10px;
      white-space: pre-wrap;
      line-height: 1.45;
      border: 1px solid transparent;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user);
      border-color: #bfdbfe;
    }
    .msg.assistant {
      align-self: flex-start;
      background: var(--assistant);
      border-color: var(--border);
    }
    .msg.assistant .md p {
      margin: 0 0 10px 0;
    }
    .msg.assistant .md p:last-child {
      margin-bottom: 0;
    }
    .msg.assistant .md ul,
    .msg.assistant .md ol {
      margin: 0 0 10px 20px;
      padding-left: 16px;
    }
    .msg.assistant .md table {
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0;
      font-size: 13px;
    }
    .msg.assistant .md th,
    .msg.assistant .md td {
      border: 1px solid var(--border);
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }
    .msg.assistant .md th {
      background: #f8fafc;
    }
    .msg.assistant .md pre {
      margin: 10px 0;
      padding: 12px;
      overflow: auto;
      border-radius: 8px;
      background: #0f172a;
      color: #e2e8f0;
      font-size: 13px;
    }
    .msg.assistant .md code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .refs {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
      font-size: 13px;
    }
    .refs a {
      color: var(--accent);
      text-decoration: none;
      word-break: break-all;
    }
    .refs a:hover {
      text-decoration: underline;
    }
    .msg.system {
      align-self: center;
      background: #f8fafc;
      color: var(--muted);
      font-size: 13px;
      max-width: 100%;
    }
    .msg.review {
      align-self: center;
      background: #fff7ed;
      color: #9a3412;
      border-color: #fdba74;
      font-size: 13px;
      max-width: 100%;
    }
    .composer {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 88px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      font: inherit;
      line-height: 1.4;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .secondary-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    button {
      border: 0;
      background: var(--accent);
      color: #fff;
      border-radius: 10px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: #e5e7eb;
      color: #111827;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .error {
      color: #b91c1c;
    }
    @media (max-width: 640px) {
      .app { padding: 14px; }
      .msg { max-width: 95%; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="app">
    <header>
      <div>
        <h1>BIA/BIS RAG Assistant</h1>
        <div class="meta">Hybrid retrieval over TestArchitect BIA/BIS docs</div>
      </div>
      <div class="meta">Open this page in a browser and share the URL on your network</div>
    </header>

    <section id="messages" class="card messages" aria-live="polite"></section>

    <section class="card composer">
      <textarea id="query" placeholder="Ask about a BIA action or BIS setting..."></textarea>
      <div class="actions">
        <div class="hint">Press Ctrl+Enter to send. Use Shift+Enter for a new line.</div>
        <div class="secondary-actions">
          <button id="escalateBtn" class="secondary" type="button" style="display:none;">Escalate to human</button>
          <button id="sendBtn" type="button">Send</button>
        </div>
      </div>
    </section>
  </main>

  <script>
    const messagesEl = document.getElementById("messages");
    const queryEl = document.getElementById("query");
    const sendBtn = document.getElementById("sendBtn");
    const escalateBtn = document.getElementById("escalateBtn");
    const history = [];
    let lastTurn = null;

    function addMessage(role, text) {
      const el = document.createElement("div");
      el.className = `msg ${role}`;
      el.textContent = text;
      messagesEl.appendChild(el);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return el;
    }

    function escapeHtml(text) {
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function renderInline(text) {
      return escapeHtml(text);
    }

    function renderMarkdown(md) {
      const lines = (md || "").replace(/\r/g, "").split("\n");
      let html = "";
      let para = [];
      let list = [];
      let table = [];
      let code = [];
      let inCode = false;

      const flushPara = () => {
        if (para.length) {
          html += `<p>${para.map(renderInline).join("<br>")}</p>`;
          para = [];
        }
      };

      const flushList = () => {
        if (list.length) {
          html += "<ul>" + list.map(item => `<li>${renderInline(item)}</li>`).join("") + "</ul>";
          list = [];
        }
      };

      const flushTable = () => {
        if (!table.length) return;
        const rows = table.slice();
        const header = rows.shift();
        if (!header) {
          table = [];
          return;
        }
        const splitRow = (row) => row
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map(cell => cell.trim());
        const headerCells = splitRow(header);
        const dataRows = rows.filter(r => !/^\s*\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?\s*$/.test(r));
        html += "<table><thead><tr>" + headerCells.map(cell => `<th>${renderInline(cell)}</th>`).join("") + "</tr></thead><tbody>";
        dataRows.forEach(row => {
          const cells = splitRow(row);
          html += "<tr>" + cells.map(cell => `<td>${renderInline(cell)}</td>`).join("") + "</tr>";
        });
        html += "</tbody></table>";
        table = [];
      };

      const flushCode = () => {
        if (code.length) {
          html += `<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`;
          code = [];
        }
      };

      for (const rawLine of lines) {
        const line = rawLine;
        const trimmed = line.trim();

        if (trimmed.startsWith("```")) {
          if (inCode) {
            flushCode();
          } else {
            flushPara();
            flushList();
            flushTable();
          }
          inCode = !inCode;
          continue;
        }

        if (inCode) {
          code.push(line);
          continue;
        }

        if (!trimmed) {
          flushPara();
          flushList();
          flushTable();
          continue;
        }

        if (/^\s*\|.*\|\s*$/.test(line)) {
          flushPara();
          flushList();
          table.push(line);
          continue;
        } else if (table.length) {
          flushTable();
        }

        if (/^[-*]\s+/.test(trimmed)) {
          flushPara();
          list.push(trimmed.replace(/^[-*]\s+/, ""));
          continue;
        } else if (list.length) {
          flushList();
        }

        para.push(trimmed);
      }

      flushPara();
      flushList();
      flushTable();
      flushCode();

      return `<div class="md">${html || "<p></p>"}</div>`;
    }

    function renderReferences(refs) {
      if (!refs || !refs.length) return "";
      const items = refs.map(ref => {
        const label = escapeHtml(ref.label || "reference");
        const url = escapeHtml(ref.url || "#");
        return `<li><a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a></li>`;
      }).join("");
      return `<div class="refs"><strong>References</strong><ul>${items}</ul></div>`;
    }

    async function sendQuery() {
      const query = queryEl.value.trim();
      if (!query) return;

      queryEl.value = "";
      addMessage("user", query);
      history.push({ role: "user", content: query });
      const pending = addMessage("assistant", "Thinking...");
      sendBtn.disabled = true;
      escalateBtn.style.display = "none";
      escalateBtn.disabled = true;
      queryEl.disabled = true;
      lastTurn = { query, answer: "", needsHumanReview: false, needsClarification: false };

      try {
        const resp = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, history }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || "Request failed");
        }
        const finalText = data.needs_clarification
          ? (data.clarifying_question || "Could you share one more detail?")
          : (data.answer || "(empty response)");
        const refsHtml = data.needs_clarification ? "" : renderReferences(data.references);
        pending.innerHTML = renderMarkdown(finalText) + refsHtml;
        history.push({ role: "assistant", content: finalText });
        lastTurn.answer = finalText;
        lastTurn.needsHumanReview = !!data.needs_human_review;
        lastTurn.needsClarification = !!data.needs_clarification;
        if (data.needs_clarification) {
          addMessage("review", data.clarifying_question ? `I need one more detail: ${data.clarifying_question}` : "I need one more detail before I can answer accurately.");
        } else if (data.needs_human_review) {
          if (data.human_review_note) {
            addMessage("review", data.human_review_note);
          }
          escalateBtn.style.display = "inline-block";
          escalateBtn.disabled = false;
        }
      } catch (err) {
        pending.className = "msg assistant error";
        pending.textContent = "Error: " + err.message;
      } finally {
        sendBtn.disabled = false;
        queryEl.disabled = false;
        queryEl.focus();
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }

    async function escalateToHuman() {
      if (!lastTurn || (!lastTurn.needsHumanReview && !lastTurn.needsClarification)) return;
      escalateBtn.disabled = true;
      try {
        const resp = await fetch("/escalate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: lastTurn.query,
            answer: lastTurn.answer,
            history,
          }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || "Escalation failed");
        }
        addMessage("review", `Escalated to human. Ticket ${data.ticket_id}.`);
        escalateBtn.style.display = "none";
      } catch (err) {
        addMessage("review", "Escalation error: " + err.message);
        escalateBtn.disabled = false;
      }
    }

    sendBtn.addEventListener("click", sendQuery);
    escalateBtn.addEventListener("click", escalateToHuman);
    queryEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.ctrlKey) {
        event.preventDefault();
        sendQuery();
      }
    });

    addMessage("system", "Ask a question to start.");
    addMessage("review", "Human review recommended for this answer.");
    queryEl.focus();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
# Serves the chat UI.
def index():
    return INDEX_HTML


# Carries a chat request payload.
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)

# Returns the chat API response shape.
class ChatResponse(BaseModel):
    answer: str
    mode: str = "answer"
    clarifying_question: str = ""
    human_review_note: str = ""
    needs_human_review: bool = False
    needs_clarification: bool = False
    references: List[Dict[str, str]] = Field(default_factory=list)

# Carries a human escalation request.
class EscalateRequest(BaseModel):
    query: str
    answer: str
    history: List[Dict[str, str]] = Field(default_factory=list)

# Returns the escalation API response shape.
class EscalateResponse(BaseModel):
    ticket_id: str
    message: str

@app.post("/chat", response_model=ChatResponse)
# Handles a chat request.
def chat(req: ChatRequest):
    try:
        result = answer_query_with_meta(req.query, history=req.history)
        return ChatResponse(
            answer=result["answer"],
            mode=result.get("mode", "answer"),
            clarifying_question=result.get("clarifying_question", ""),
            human_review_note=result.get("human_review_note", ""),
            needs_human_review=result["needs_human_review"],
            needs_clarification=result.get("needs_clarification", False),
            references=result.get("references", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer query: {exc}") from exc

@app.post("/escalate", response_model=EscalateResponse)
# Records a human escalation ticket.
def escalate(req: EscalateRequest):
    ticket_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    print(
        f"[escalate] ticket={ticket_id} ts={timestamp} query={req.query!r}",
        file=sys.stderr,
    )
    return EscalateResponse(
        ticket_id=ticket_id,
        message="Escalation recorded for human review.",
    )

# Starts the FastAPI server.
def main():
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        lifespan="off",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
