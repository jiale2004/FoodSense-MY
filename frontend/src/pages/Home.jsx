import { Link } from "react-router-dom";
import Flourish from "../components/Flourish.jsx";
import HeroVisual from "../components/HeroVisual.jsx";
import NasiLemakIcon from "../components/NasiLemakIcon.jsx";

const DISHES = [
  { key: "nasi_lemak", label: "Nasi Lemak", icon: "🍛" },
  { key: "roti_canai", label: "Roti Canai", icon: "🫓" },
  { key: "char_kuey_teow", label: "Char Kuey Teow", icon: "🍝" },
  { key: "chicken_rice", label: "Chicken Rice", icon: "🍗" },
  { key: "laksa", label: "Laksa", icon: "🍲" },
  { key: "mee_goreng", label: "Mee Goreng", icon: "🍜" },
];

const PROBLEMS = [
  {
    icon: "📝",
    title: "Manual logging doesn't scale",
    text: "Food diaries and calorie logs rely on memory and self-reporting, which is slow and prone to underreporting.",
  },
  {
    icon: "🌏",
    title: "Global apps carry Western bias",
    text: "Studies show mainstream nutrition apps systematically underestimate the energy and macronutrients of complex Asian dishes by an average of 1,520 kJ.",
  },
  {
    icon: "🍽️",
    title: "Classification isn't enough",
    text: "Most Malaysian food datasets only support whole-image classification, so they fail the moment two dishes share a single plate.",
  },
];

const STEPS = [
  {
    icon: "📷",
    title: "Capture a photo",
    text: "Snap or upload a picture of your plate — single dish or several mixed together.",
  },
  {
    icon: "🎯",
    title: "Detect & localize",
    text: "A YOLO-based object detector draws bounding boxes around every dish it recognizes, with a confidence score for each.",
  },
  {
    icon: "📚",
    title: "Match a knowledge base",
    text: "Each detected class is linked to a structured nutrition and allergen reference — not a guess, a lookup.",
  },
  {
    icon: "💬",
    title: "Explain in plain language",
    text: "A bounded LLM turns the structured result into a readable, advisory-only summary grounded in the knowledge base.",
  },
];

const METRICS = [
  { value: "94.0%", label: "Precision" },
  { value: "86.8%", label: "Recall" },
  { value: "88.6%", label: "mAP@0.5" },
  { value: "67.3%", label: "mAP@0.5:0.95" },
];

export default function Home() {
  return (
    <div className="home-shell">
      <nav className="home-nav">
        <div className="home-nav-inner">
          <span className="nav-brand">
            <NasiLemakIcon size={24} /> FoodSense<b>MY</b>
          </span>
          <Link to="/scan" className="btn btn-secondary nav-cta">
            Try It
          </Link>
        </div>
      </nav>

      <main className="page home-page">
        <section className="hero">
          <div className="hero-grid">
            <div className="hero-copy">
              <div className="hero-badge">CSC3014 Computer Vision · Group 9 · Sunway University</div>
              <h1>
                See what's on the plate.
                <br />
                <span className="brand-accent">Understand what it means.</span>
              </h1>
              <p className="hero-subtitle">
                FoodSense-MY is a computer vision system that detects multiple
                Malaysian dishes in a single photo and turns each one into a
                structured, plain-language nutritional advisory — built to
                close the regional gap left by global dietary-tracking apps.
              </p>
              <div className="hero-actions">
                <Link to="/scan" className="btn btn-hero-primary">
                  📷 Try Scanning a Photo
                </Link>
                <a href="#how-it-works" className="btn btn-hero-secondary">
                  See how it works
                </a>
              </div>
            </div>
            <HeroVisual />
          </div>
        </section>

        <section className="section">
          <h2 className="section-title">Why this matters</h2>
          <Flourish />
          <div className="problem-grid">
            {PROBLEMS.map((p) => (
              <div className="problem-card" key={p.title}>
                <span className="problem-icon" aria-hidden="true">{p.icon}</span>
                <h3>{p.title}</h3>
                <p>{p.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section" id="how-it-works">
          <h2 className="section-title">How it works</h2>
          <Flourish />
          <div className="steps">
            {STEPS.map((s, i) => (
              <div className="step-card" key={s.title}>
                <div className="step-number">{i + 1}</div>
                <span className="step-icon" aria-hidden="true">{s.icon}</span>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section-title">Trained on six Malaysian favourites</h2>
          <Flourish />
          <p className="section-subtitle">
            Every class was built from a locally curated, bounding-box-annotated
            dataset — not a repurposed Western food set.
          </p>
          <div className="dish-grid">
            {DISHES.map((d) => (
              <div className="dish-chip" key={d.key}>
                <span className="dish-icon-badge" aria-hidden="true">{d.icon}</span>
                <span>{d.label}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="section metrics-section">
          <h2 className="section-title">Locked test-set performance</h2>
          <Flourish />
          <p className="section-subtitle">
            Interim v8_n_mg detector, evaluated once on a held-out test split
            (82 images, 84 boxes) — thresholds calibrated on validation only.
          </p>
          <div className="metrics-grid">
            {METRICS.map((m) => (
              <div className="metric-card" key={m.label}>
                <div className="metric-value">{m.value}</div>
                <div className="metric-label">{m.label}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="cta-banner">
          <h2>Ready to try it yourself?</h2>
          <p>Upload a plate and see the detector in action.</p>
          <Link to="/scan" className="btn btn-cta-inverse">
            📷 Try Scanning a Photo
          </Link>
        </section>

        <footer className="page-footer">
          <p>Built for CSC3014 Computer Vision, Group 9 — Sunway University.</p>
          <p>For informational purposes only — not medical advice.</p>
        </footer>
      </main>
    </div>
  );
}
