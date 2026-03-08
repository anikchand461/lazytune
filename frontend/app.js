/* ── Model metadata ─────────────────────────── */
const MODEL_METRICS = {
  RandomForestClassifier: ["accuracy", "f1", "precision", "recall"],
  SVC: ["accuracy", "f1", "precision", "recall"],
  LogisticRegression: ["accuracy", "f1", "precision", "recall"],
  RandomForestRegressor: ["r2", "neg_mean_squared_error"],
  LinearRegression: ["r2", "neg_mean_squared_error"],
};
const MODEL_PARAMS = {
  RandomForestClassifier: [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
  ],
  RandomForestRegressor: [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
  ],
  SVC: ["C", "kernel", "degree", "gamma", "coef0", "shrinking"],
  LogisticRegression: [
    "C",
    "solver",
    "penalty",
    "max_iter",
    "fit_intercept",
    "class_weight",
  ],
  LinearRegression: ["fit_intercept", "copy_X", "positive"],
};
const PARAM_HINTS = {
  n_estimators: "example: 50,100,200",
  max_depth: "example: 5,10,20 or null",
  min_samples_split: "example: 2,5,10",
  min_samples_leaf: "example: 1,2,4",
  max_features: "options: sqrt,log2 or number",
  bootstrap: "options: true,false",
  criterion:
    "classifier: gini,entropy | regressor: squared_error,absolute_error",
  C: "example: 0.01,0.1,1,10",
  kernel: "options: linear,rbf,poly,sigmoid",
  degree: "example: 2,3,4 (poly kernel)",
  gamma: "options: scale,auto or 0.1",
  coef0: "example: 0,0.1,0.5",
  shrinking: "options: true,false",
  solver: "options: lbfgs,liblinear,newton-cg,saga",
  penalty: "options: l1,l2,elasticnet,none",
  max_iter: "example: 100,200,500",
  fit_intercept: "options: true,false",
  class_weight: "options: balanced or none",
  copy_X: "options: true,false",
  positive: "options: true,false",
};

/* ── Loader ─────────────────────────────────── */
window.addEventListener("load", () => {
  setTimeout(() => {
    const loader = document.getElementById("loader");
    if (loader) loader.classList.add("hidden");
  }, 1800);
});

/* ── Particle canvas ────────────────────────── */
(function initParticles() {
  const canvas = document.getElementById("particles");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H;

  const isMobile = () => window.innerWidth < 600;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  const COUNT = isMobile() ? 30 : 60;
  const pts = Array.from({ length: COUNT }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    r: Math.random() * 1.3 + 0.4,
  }));

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const linkDist = isMobile() ? 100 : 140;

    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x,
          dy = pts[i].y - pts[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < linkDist) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(200,255,0,${0.12 * (1 - d / linkDist)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    pts.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(200,255,0,0.35)";
      ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── Toast ──────────────────────────────────── */
function toast(msg, type = "ok") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast ${type} show`;
  clearTimeout(el._t);
  el._t = setTimeout(() => {
    el.className = "toast";
  }, 3000);
}

/* ── Status ─────────────────────────────────── */
function setStatus(state, label) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (!dot || !text) return;
  text.textContent = label;
  const map = {
    ready: ["#c8ff00", "0 0 10px rgba(200,255,0,.6)"],
    running: ["#facc15", "0 0 10px rgba(250,204,21,.6)"],
    done: ["#00e5ff", "0 0 10px rgba(0,229,255,.6)"],
    error: ["#ff3c3c", "0 0 10px rgba(255,60,60,.6)"],
  };
  const [bg, sh] = map[state] || map.ready;
  dot.style.background = bg;
  dot.style.boxShadow = sh;
}

/* ── Stepper ─────────────────────────────────── */
function setStep(n) {
  document.querySelectorAll(".step").forEach((s) => {
    const num = +s.dataset.step;
    s.classList.remove("active", "done");
    if (num === n) s.classList.add("active");
    else if (num < n) s.classList.add("done");
  });
}

/* ── updateOptions ──────────────────────────── */
function updateOptions() {
  const model = document.getElementById("model").value;
  const ms = document.getElementById("metric");
  ms.innerHTML = "";
  MODEL_METRICS[model].forEach((m) => {
    const o = document.createElement("option");
    o.value = m;
    o.text = m;
    ms.appendChild(o);
  });
  document.getElementById("params-container").innerHTML = "";
  addParam();
}

/* ── addParam ───────────────────────────────── */
function addParam() {
  const model = document.getElementById("model").value;
  const cont = document.getElementById("params-container");
  const row = document.createElement("div");
  row.className = "param-row";

  const sw = document.createElement("div");
  sw.className = "select-wrap";
  const sel = document.createElement("select");
  MODEL_PARAMS[model].forEach((p) => {
    const o = document.createElement("option");
    o.value = p;
    o.text = p;
    sel.appendChild(o);
  });
  const chev = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  chev.setAttribute("viewBox", "0 0 24 24");
  chev.setAttribute("fill", "none");
  chev.setAttribute("stroke", "currentColor");
  chev.setAttribute("stroke-width", "2");
  chev.setAttribute("stroke-linecap", "round");
  chev.classList.add("select-chevron");
  const poly = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "polyline",
  );
  poly.setAttribute("points", "6 9 12 15 18 9");
  chev.appendChild(poly);
  sw.append(sel, chev);

  const inp = document.createElement("input");
  inp.type = "text";
  inp.placeholder = PARAM_HINTS[sel.value] || "comma separated values";
  sel.onchange = () => {
    inp.placeholder = PARAM_HINTS[sel.value] || "comma separated values";
  };

  const rm = document.createElement("button");
  rm.innerHTML = "&times;";
  rm.className = "btn-remove";
  rm.onclick = () => row.remove();

  row.append(sw, inp, rm);
  cont.appendChild(row);
}

/* ── buildParamGrid ─────────────────────────── */
function buildParamGrid() {
  const grid = {};
  document.querySelectorAll(".param-row").forEach((row) => {
    const param = row.querySelector("select").value;
    const values = row
      .querySelector("input")
      .value.split(",")
      .map((v) => v.trim())
      .filter(Boolean)
      .map((v) =>
        v === "true" ? true : v === "false" ? false : isNaN(v) ? v : Number(v),
      );
    if (values.length) grid[param] = values;
  });
  return grid;
}

/* ── uploadDataset ──────────────────────────── */
async function uploadDataset() {
  const file = document.getElementById("dataset").files[0];
  if (!file) {
    toast("select a file first", "err");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  try {
    await fetch("http://127.0.0.1:8000/datasets/upload", {
      method: "POST",
      body: fd,
    });
    toast("dataset uploaded");
    setStep(2);
  } catch {
    toast("upload failed", "err");
  }
}

/* ── runOptimization ────────────────────────── */
async function runOptimization() {
  const model = document.getElementById("model").value;
  const target = document.getElementById("target").value;
  const metric = document.getElementById("metric").value;
  const param_grid = buildParamGrid();
  const result = document.getElementById("result");
  result.textContent = "Running optimization...";
  result.style.color = "#555";
  setStatus("running", "Running...");
  setStep(4);
  try {
    const res = await fetch("http://127.0.0.1:8000/optimize/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, target, metric, param_grid }),
    });
    const data = await res.json();
    result.textContent = JSON.stringify(data, null, 2);
    result.style.color = "var(--accent)";
    document.getElementById("btn-copy").style.display = "inline-flex";
    toast("optimization complete");
    setStatus("done", "Done");
    document
      .getElementById("section-results")
      .scrollIntoView({ behavior: "smooth" });
  } catch {
    result.textContent = "Backend error — is the server running?";
    result.style.color = "var(--danger)";
    toast("optimization failed", "err");
    setStatus("error", "Error");
  }
}

/* ── copyResult ─────────────────────────────── */
function copyResult() {
  navigator.clipboard.writeText(document.getElementById("result").textContent);
  toast("copied to clipboard");
}

/* ── DOMContentLoaded ───────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  updateOptions();

  // file label
  const inp = document.getElementById("dataset");
  const lbl = document.getElementById("file-label");
  if (inp && lbl) {
    inp.addEventListener("change", () => {
      lbl.textContent = inp.files[0] ? inp.files[0].name : "No file chosen";
    });
  }

  // drag & drop on upload zone
  const zone = document.getElementById("upload-zone");
  if (zone) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.style.borderColor = "var(--accent)";
    });
    zone.addEventListener("dragleave", () => {
      zone.style.borderColor = "";
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.style.borderColor = "";
      const f = e.dataTransfer.files[0];
      if (f && lbl) {
        const dt = new DataTransfer();
        dt.items.add(f);
        inp.files = dt.files;
        lbl.textContent = f.name;
      }
    });
  }

  // stepper nav
  const sectionMap = { 1: "dataset", 2: "model", 3: "params", 4: "results" };
  document.querySelectorAll(".step").forEach((s) => {
    s.addEventListener("click", () => {
      const n = +s.dataset.step;
      setStep(n);
      const el = document.getElementById("section-" + sectionMap[n]);
      if (el) el.scrollIntoView({ behavior: "smooth" });
    });
  });
});
