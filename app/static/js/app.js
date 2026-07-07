const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");
const results = document.getElementById("results");
const detectionsEl = document.getElementById("detections");
const nutritionEl = document.getElementById("nutrition");
const advisoryText = document.getElementById("advisory-text");
const processingTime = document.getElementById("processing-time");

let selectedFile = null;

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function setPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.remove("hidden");
    dropZone.querySelector(".drop-zone-content").classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please select a valid image file (JPEG, PNG, or WebP).");
    return;
  }
  hideError();
  selectedFile = file;
  setPreview(file);
  analyzeBtn.disabled = false;
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
});

function renderDetections(detections) {
  detectionsEl.innerHTML = "";
  if (!detections.length) {
    detectionsEl.innerHTML = '<p class="hint">No dishes detected. Try a clearer photo.</p>';
    return;
  }

  detections.forEach((d) => {
    const pct = Math.round(d.confidence * 100);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <h4>${formatName(d.class_name)}</h4>
      <p>Confidence: ${pct}%</p>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width: ${pct}%"></div>
      </div>
    `;
    detectionsEl.appendChild(card);
  });
}

function renderNutrition(entries) {
  nutritionEl.innerHTML = "";
  if (!entries.length) return;

  entries.forEach((n) => {
    const card = document.createElement("div");
    card.className = "card";

    let macros = "";
    if (n.found_in_kb) {
      macros = `
        <div class="macro-row"><span>Calories</span><span>${n.calories_kcal ?? "—"} kcal</span></div>
        <div class="macro-row"><span>Protein</span><span>${n.protein_g ?? "—"} g</span></div>
        <div class="macro-row"><span>Carbs</span><span>${n.carbs_g ?? "—"} g</span></div>
        <div class="macro-row"><span>Fat</span><span>${n.fat_g ?? "—"} g</span></div>
        <div class="macro-row"><span>Sodium</span><span>${n.sodium_mg ?? "—"} mg</span></div>
      `;
    } else {
      macros = '<p class="hint">No verified data in knowledge base.</p>';
    }

    let tags = "";
    n.dietary_tags.forEach((t) => {
      tags += `<span class="tag">${t}</span>`;
    });
    n.allergens.forEach((a) => {
      tags += `<span class="tag tag-allergen">${a}</span>`;
    });

    card.innerHTML = `
      <h4>${n.display_name}</h4>
      ${macros}
      ${tags ? `<div style="margin-top:0.5rem">${tags}</div>` : ""}
      ${n.health_notes ? `<p class="hint" style="margin-top:0.5rem">${n.health_notes}</p>` : ""}
    `;
    nutritionEl.appendChild(card);
  });
}

function formatName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  results.classList.add("hidden");
  loading.classList.remove("hidden");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();

    renderDetections(data.detections);
    renderNutrition(data.nutrition);
    advisoryText.textContent = data.advisory_text;
    processingTime.textContent = `Processed in ${data.processing_ms} ms`;

    results.classList.remove("hidden");
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    loading.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});
