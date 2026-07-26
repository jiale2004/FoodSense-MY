import { useEffect, useRef, useState } from "react";

function ChatIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

const WELCOME =
  "Hi! Ask me anything about Malaysian dishes, calories, allergens, or the app — no photo needed. If you upload a meal later, I can also answer about that scan.";

const SUGGESTIONS = [
  "How many calories in nasi lemak?",
  "Is laksa high in sodium?",
  "What allergens are in roti canai?",
  "Difference between mee goreng and char kuey teow?",
];

export default function ChatWidget({ mealContext }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", content: WELCOME },
  ]);
  const listRef = useRef(null);
  const inputRef = useRef(null);
  const hasScan = Boolean(mealContext?.detections?.length);

  useEffect(() => {
    if (!listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, open, busy]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function sendMessage(rawText) {
    const text = (rawText ?? input).trim();
    if (!text || busy) return;

    const nextHistory = [...messages, { role: "user", content: text }];
    setMessages(nextHistory);
    setInput("");
    setBusy(true);
    setError("");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: nextHistory
            .filter((m) => m.role === "user" || m.role === "assistant")
            .slice(0, -1)
            .slice(-12),
          context: mealContext || null,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
              : `Chat failed (${res.status})`;
        throw new Error(msg);
      }
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply || "No reply received." },
      ]);
    } catch (err) {
      setError(err.message || "Chat unavailable. Try again.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry — I could not reach the assistant. Is the backend running?",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-widget">
      {open && (
        <section className="chat-panel" aria-label="FoodSense AI chat">
          <header className="chat-header">
            <div>
              <strong>Ask FoodSense</strong>
              <p>
                {hasScan
                  ? "Using your latest scan + knowledge base"
                  : "Ask anytime — photo optional"}
              </p>
            </div>
            <button
              type="button"
              className="chat-icon-btn"
              aria-label="Close chat"
              onClick={() => setOpen(false)}
            >
              <CloseIcon />
            </button>
          </header>

          <div className="chat-messages" ref={listRef}>
            {messages.map((m, i) => (
              <div
                key={i}
                className={`chat-bubble ${m.role === "user" ? "is-user" : "is-assistant"}`}
              >
                {m.content}
              </div>
            ))}
            {messages.length === 1 && !busy && (
              <div className="chat-suggestions">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="chat-suggestion"
                    onClick={() => sendMessage(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            {busy && (
              <div className="chat-bubble is-assistant is-typing">
                <span />
                <span />
                <span />
              </div>
            )}
          </div>

          {error && <p className="chat-error">{error}</p>}

          <form
            className="chat-compose"
            onSubmit={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about Malaysian food…"
              disabled={busy}
              maxLength={2000}
            />
            <button
              type="submit"
              className="chat-send"
              disabled={busy || !input.trim()}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </form>
        </section>
      )}

      <button
        type="button"
        className={`chat-fab${open ? " is-open" : ""}`}
        aria-label={open ? "Close chat" : "Open AI chat"}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <CloseIcon /> : <ChatIcon />}
      </button>
    </div>
  );
}
