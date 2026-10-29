const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const resultEl = document.getElementById("result");
const predictedClassEl = document.getElementById("predictedClass");
const confidenceEl = document.getElementById("confidence");
const probabilitiesEl = document.getElementById("probabilities");
const errorBox = document.getElementById("errorBox");
const loader = document.getElementById("loader");
const clearBtn = document.getElementById("clearBtn");

let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
    showError("Please select a JPEG or PNG image.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  dropzoneContent.hidden = true;
  predictBtn.disabled = false;
  clearBtn.hidden = false;
  hideResult();
  hideError();
}

clearBtn.addEventListener("click", resetState);

function resetState() {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  preview.hidden = true;
  dropzoneContent.hidden = false;
  predictBtn.disabled = true;
  clearBtn.hidden = true;
  hideResult();
  hideError();
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  hideResult();
  loader.hidden = false;
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Something went wrong while analyzing the image.");
    }
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    loader.hidden = true;
    predictBtn.disabled = false;
  }
});

function renderResult(data) {
  predictedClassEl.textContent = data.predicted_class;
  confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const sorted = [...data.probabilities].sort((a, b) => b.probability - a.probability);

  probabilitiesEl.innerHTML = "";
  sorted.forEach((p) => {
    const isTop = p.class_name === data.predicted_class;
    const row = document.createElement("div");
    row.className = `prob-row${isTop ? " top" : ""}`;
    row.innerHTML = `
      <span class="prob-name">${p.class_name}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:${(p.probability * 100).toFixed(1)}%"></div>
      </div>
      <span class="prob-value">${(p.probability * 100).toFixed(1)}%</span>
    `;
    probabilitiesEl.appendChild(row);
  });

  resultEl.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hideError() { errorBox.hidden = true; }
function hideResult() { resultEl.hidden = true; }