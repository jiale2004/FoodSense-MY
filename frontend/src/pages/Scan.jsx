import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import NasiLemakIcon from "../components/NasiLemakIcon.jsx";

const formatName = (name) =>
  name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const confidenceTier = (confidence) => {
  if (confidence >= 0.75) return "high";
  if (confidence >= 0.5) return "medium";
  return "low";
};

function UploadIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function RemoveIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

export default function Scan() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFile(selected) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG, PNG, or WebP).");
      return;
    }
    setError("");
    setResult(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  function clearFile() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <Link to="/" className="back-link">← Back to Home</Link>

      <header className="app-header">
        <div className="brand-icon">
          <NasiLemakIcon size={32} />
        </div>
        <h1>
          FoodSense<span className="brand-accent">MY</span>
        </h1>
        <p className="subtitle">
          Upload a photo of your Malaysian meal and get instant dish detection
          with nutritional insight.
        </p>
      </header>

      <section
        className={`upload-card${isDragging ? " is-dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        {previewUrl ? (
          <div className="preview-wrap">
            <img className="preview" src={previewUrl} alt="Selected preview" />
            <button
              type="button"
              className="remove-btn"
              aria-label="Remove image"
              onClick={(e) => {
                e.stopPropagation();
                clearFile();
              }}
            >
              <RemoveIcon />
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <div className="upload-icon-wrap">
              <UploadIcon />
            </div>
            <p className="drop-title">Drag &amp; drop a food photo here</p>
            <p className="hint">or</p>
            <span className="btn btn-secondary">Choose File</span>
            <p className="hint small">JPEG, PNG, or WebP · max 10 MB</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </section>

      <button className="analyze-btn" disabled={!file || loading} onClick={analyze}>
        {loading ? (
          <>
            <span className="spinner" /> Analyzing…
          </>
        ) : (
          "Analyze Food"
        )}
      </button>

      {error && <p className="error-banner">⚠️ {error}</p>}

      {result && (
        <section className="results">
          <span className="meta">Processed in {result.processing_ms} ms</span>

          <h2>Detected Dishes</h2>
          {result.detections.length === 0 ? (
            <p className="hint">No dishes detected. Try a clearer photo.</p>
          ) : (
            <ul className="cards">
              {result.detections.map((d, i) => {
                const pct = Math.round(d.confidence * 100);
                const tier = confidenceTier(d.confidence);
                return (
                  <li className="card detection-card" key={i}>
                    <div className="detection-head">
                      <strong>{formatName(d.class_name)}</strong>
                      <span className={`confidence-badge tier-${tier}`}>{pct}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div className="confidence-fill" style={{ width: `${pct}%` }} />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}

          {result.nutrition.length > 0 && (
            <>
              <h2>Nutrition Facts</h2>
              <ul className="cards nutrition-grid">
                {result.nutrition.map((n, i) => (
                  <li className="card nutrition-card" key={i}>
                    <div className="nutrition-head">
                      <strong>{n.display_name}</strong>
                    </div>
                    {n.found_in_kb ? (
                      <>
                        <div className="calorie-figure">
                          {n.calories_kcal ?? "—"}
                          <span>kcal</span>
                        </div>
                        <div className="macro-row">
                          <span className="macro-chip protein">P {n.protein_g ?? "—"} g</span>
                          <span className="macro-chip carbs">C {n.carbs_g ?? "—"} g</span>
                          <span className="macro-chip fat">F {n.fat_g ?? "—"} g</span>
                        </div>
                      </>
                    ) : (
                      <p className="hint">No verified data in knowledge base.</p>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}

          {result.advisory_text && (
            <>
              <h2>Advisory</h2>
              <div className="advisory-card">
                <span className="advisory-icon" aria-hidden="true">💡</span>
                <div>
                  <p>{result.advisory_text}</p>
                </div>
              </div>
            </>
          )}

          {result.disclaimer && <p className="disclaimer">{result.disclaimer}</p>}
        </section>
      )}

      <footer className="page-footer">
        For informational purposes only — not medical advice.
      </footer>
    </main>
  );
}
