import { useRef, useState } from "react";

const formatName = (name) =>
  name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
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
    <main className="container">
      <header>
        <h1>FoodSense-MY</h1>
        <p className="subtitle">Upload a food image to test detection.</p>
      </header>

      <section
        className="drop-zone"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        {previewUrl ? (
          <img className="preview" src={previewUrl} alt="Selected preview" />
        ) : (
          <p>Click or drag an image here</p>
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
        {loading ? "Analyzing…" : "Analyze"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="results">
          <p className="meta">Processed in {result.processing_ms} ms</p>

          <h2>Detections</h2>
          {result.detections.length === 0 ? (
            <p className="hint">No dishes detected. Try a clearer photo.</p>
          ) : (
            <ul className="cards">
              {result.detections.map((d, i) => (
                <li className="card" key={i}>
                  <strong>{formatName(d.class_name)}</strong>
                  <span>{Math.round(d.confidence * 100)}%</span>
                </li>
              ))}
            </ul>
          )}

          {result.nutrition.length > 0 && (
            <>
              <h2>Nutrition</h2>
              <ul className="cards">
                {result.nutrition.map((n, i) => (
                  <li className="card nutrition" key={i}>
                    <strong>{n.display_name}</strong>
                    {n.found_in_kb ? (
                      <div className="macros">
                        <span>{n.calories_kcal ?? "—"} kcal</span>
                        <span>P {n.protein_g ?? "—"} g</span>
                        <span>C {n.carbs_g ?? "—"} g</span>
                        <span>F {n.fat_g ?? "—"} g</span>
                      </div>
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
              <p className="advisory">{result.advisory_text}</p>
            </>
          )}

          {result.disclaimer && <p className="disclaimer">{result.disclaimer}</p>}
        </section>
      )}
    </main>
  );
}
