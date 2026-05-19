from rag.chat import answer_query_with_meta

# Runs the interactive CLI chat loop.
def main():
    print("BIA/BIS Assistant. Type 'exit' to quit.")
    history = []
    try:
        while True:
            try:
                q = input("You: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break

            if not q or q.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            try:
                result = answer_query_with_meta(q, history=history)
                if result.get("needs_clarification"):
                    msg = result.get("clarifying_question") or "Could you share one more detail?"
                else:
                    msg = result["answer"]
                print("\nAssistant:\n", msg, "\n")
                refs = [] if result.get("needs_clarification") else (result.get("references") or [])
                if refs:
                    print("References:")
                    for ref in refs[:3]:
                        label = ref.get("label") or "reference"
                        url = ref.get("url") or ""
                        print(f"- {label}: {url}")
                    print()
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": msg})
                if result.get("needs_human_review"):
                    print("[review] Human review recommended for this answer.\n")
                if result.get("needs_clarification"):
                    print("[clarify] Please answer the follow-up so I can refine the result.\n")
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye.")
                break
            except Exception as exc:
                print(f"\n[error] {exc}\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")

if __name__ == "__main__":
    main()
