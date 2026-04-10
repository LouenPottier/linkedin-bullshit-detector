// popup.js — v0.5
// Inclut un entraîneur Ridge in-browser (vocab TF-IDF fixe)

const STORAGE_KEYS = [
  "bullshit_dataset", "bsd_mode", "bsd_threshold",
  "bsd_hide_sponsored", "bsd_silent_hide", "bsd_custom_model"
];
const MIN_POSTS_TO_TRAIN = 20;
// Alpha adaptatif : régularisation forte sur petits datasets, faible sur grands
// Calibré pour correspondre à sklearn sur ~200 posts (alpha=0.1)
// < 30 posts  → 10.0  (très forte régularisation)
// ~ 50 posts  →  2.0
// ~100 posts  →  0.5
// ~200 posts  →  0.1  (valeur sklearn par défaut)
// ~500 posts  →  0.02
function adaptiveAlpha(n) {
  // Interpolation log-linéaire : alpha = k / n
  // Calibrée pour que alpha(200) ≈ 0.1 → k = 20
  return Math.max(20 / n, 0.01);
}

// ============================================================
// TF-IDF (vocab fixe depuis tfidf_vocab.json)
// ============================================================

async function loadVocab() {
  const url = chrome.runtime.getURL("tfidf_vocab.json");
  const res = await fetch(url);
  return res.json();
}

function tokenize(text) {
  return (text || "").toLowerCase().match(/[a-z0-9\u00c0-\u017e]+/g) || [];
}

function tfidfVector(text, vocab, idf, nFeatures) {
  const tokens = tokenize(text);
  const tf = new Float64Array(nFeatures);

  for (const tok of tokens) {
    if (tok in vocab) tf[vocab[tok]] += 1;
  }
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = tokens[i] + " " + tokens[i + 1];
    if (bigram in vocab) tf[vocab[bigram]] += 1;
  }
  for (let i = 0; i < nFeatures; i++) {
    if (tf[i] > 0) tf[i] = 1 + Math.log(tf[i]);
  }
  for (let i = 0; i < nFeatures; i++) tf[i] *= idf[i];
  let norm = 0;
  for (let i = 0; i < nFeatures; i++) norm += tf[i] * tf[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < nFeatures; i++) tf[i] /= norm;

  return tf;
}


// Recalcule les IDF sur le corpus fourni (vocab fixe, smooth_idf=True comme sklearn)
// idf(t) = log((1 + n) / (1 + df(t))) + 1
function computeIDF(texts, vocab, nFeatures) {
  const n  = texts.length;
  const df = new Float64Array(nFeatures);

  for (const text of texts) {
    const tokens = tokenize(text);
    const seen   = new Set();
    // Unigrammes
    for (const tok of tokens) {
      if (tok in vocab && !seen.has(vocab[tok])) {
        df[vocab[tok]]++;
        seen.add(vocab[tok]);
      }
    }
    // Bigrammes
    for (let i = 0; i < tokens.length - 1; i++) {
      const bigram = tokens[i] + " " + tokens[i + 1];
      if (bigram in vocab && !seen.has(vocab[bigram])) {
        df[vocab[bigram]]++;
        seen.add(vocab[bigram]);
      }
    }
  }

  const idf = new Float64Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) {
    idf[i] = Math.log((1 + n) / (1 + df[i])) + 1;
  }
  return idf;
}

// ============================================================
// Features numériques
// ============================================================

function parseCount(s) {
  const d = String(s || "").replace(/[^\d]/g, "");
  return d ? parseInt(d, 10) : 0;
}

function fcFlag(fc) {
  return fc && fc.toLowerCase().includes("profil") ? 1 : 0;
}

function buildNumFeatures(post) {
  const text      = post.text || "";
  const headline  = post.headline || "";
  const wordCount = text.split(/\s+/).length;
  const emojiCount = [...text].filter(c => c.codePointAt(0) > 0x1F000).length;
  return [
    parseCount(post.likes),
    parseCount(post.comments),
    post.autoScore || 0,
    text.length,
    wordCount,
    fcFlag(post.feedContext),
    (post.autoKeywords || []).length,
    emojiCount / Math.max(wordCount, 1),
    headline.length,
  ];
}

// ============================================================
// StandardScaler
// ============================================================

function fitScaler(matrix) {
  const n = matrix[0].length;
  const mean  = new Float64Array(n);
  const scale = new Float64Array(n);

  for (const row of matrix) {
    for (let j = 0; j < n; j++) mean[j] += row[j];
  }
  for (let j = 0; j < n; j++) mean[j] /= matrix.length;

  for (const row of matrix) {
    for (let j = 0; j < n; j++) scale[j] += (row[j] - mean[j]) ** 2;
  }
  // ddof=1 comme sklearn.StandardScaler
  const ddof = Math.max(matrix.length - 1, 1);
  for (let j = 0; j < n; j++) {
    scale[j] = Math.sqrt(scale[j] / ddof) || 1;
  }
  return { mean, scale };
}

function applyScaler(row, scaler) {
  return row.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]);
}

// ============================================================
// Ridge — résolution analytique exacte : w = (XᵀX + αI)⁻¹ Xᵀy
// Identique à sklearn.Ridge, via élimination de Gauss (Cholesky-free)
// ============================================================

// Produit matrice-vecteur
function matVec(A, v, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      out[i] += A[i * n + j] * v[j];
  return out;
}

// Résolution du système (XᵀX + αI) w = Xᵀy par élimination de Gauss avec pivot partiel
function solveLinear(A, b, n) {
  // Copie pour ne pas muter A
  const M = new Float64Array(A);
  const x = new Float64Array(b);

  for (let col = 0; col < n; col++) {
    // Pivot partiel
    let maxVal = Math.abs(M[col * n + col]), maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(M[row * n + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    // Swap lignes
    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        const tmp = M[col * n + k]; M[col * n + k] = M[maxRow * n + k]; M[maxRow * n + k] = tmp;
      }
      const tmp = x[col]; x[col] = x[maxRow]; x[maxRow] = tmp;
    }
    const pivot = M[col * n + col];
    if (Math.abs(pivot) < 1e-12) continue;
    for (let row = col + 1; row < n; row++) {
      const factor = M[row * n + col] / pivot;
      for (let k = col; k < n; k++) M[row * n + k] -= factor * M[col * n + k];
      x[row] -= factor * x[col];
    }
  }
  // Substitution arrière
  const w = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= M[i * n + j] * w[j];
    w[i] = s / M[i * n + i];
  }
  return w;
}

// Ridge analytique avec centrage (équivalent sklearn avec fit_intercept=True)
// X : tableau de tableaux Float64, y : tableau de nombres
function solveRidge(X, y, alpha) {
  const m = X.length;
  const n = X[0].length;

  // Centrer X et y (équivaut à fit_intercept=True de sklearn)
  const xMean = new Float64Array(n);
  let yMean = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) xMean[j] += X[i][j];
    yMean += y[i];
  }
  for (let j = 0; j < n; j++) xMean[j] /= m;
  yMean /= m;

  const Xc = X.map(row => row.map((v, j) => v - xMean[j]));
  const yc  = y.map(v => v - yMean);

  // Construire XᵀX (n×n) + αI
  const XtX = new Float64Array(n * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++)
        XtX[j * n + k] += Xc[i][j] * Xc[i][k];
  for (let j = 0; j < n; j++) XtX[j * n + j] += alpha;

  // Construire Xᵀy (vecteur n)
  const Xty = new Float64Array(n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      Xty[j] += Xc[i][j] * yc[i];

  // Résoudre (XᵀX + αI) w = Xᵀy
  const w = solveLinear(XtX, Xty, n);

  // Intercept : yMean - xMean·w
  let intercept = yMean;
  for (let j = 0; j < n; j++) intercept -= xMean[j] * w[j];

  return { coef: Array.from(w), intercept };
}

function computeMAE(X, y, coef, intercept) {
  let mae = 0;
  for (let i = 0; i < X.length; i++) {
    let pred = intercept;
    for (let j = 0; j < coef.length; j++) pred += coef[j] * X[i][j];
    pred = Math.max(0, Math.min(10, pred));
    mae += Math.abs(pred - y[i]);
  }
  return mae / X.length;
}

// Hash déterministe d'un postId (djb2) → entier positif
function hashPostId(id) {
  let h = 5381;
  for (let i = 0; i < (id || "").length; i++) {
    h = ((h << 5) + h) ^ (id || "").charCodeAt(i);
    h = h | 0; // keep 32-bit
  }
  return Math.abs(h);
}

// ============================================================
// Pipeline d'entraînement complet
// ============================================================

async function trainModel(posts, vocabData, onProgress) {
  const vocab  = vocabData.vocabulary;
  const nTfidf = vocabData.n_tfidf_features;
  const nNum   = vocabData.n_num_features;

  // Recalculer les IDF sur le corpus labellisé (vocab fixe, smooth_idf=True)
  const labelledPosts = posts.filter(p => p.manualScore !== null && p.manualScore !== undefined);
  const corpus = labelledPosts.map(p => `${p.text || ""} ${p.headline || ""}`.trim());
  const idf    = computeIDF(corpus, vocab, nTfidf);

  // Construire toutes les features
  const Xraw   = [];
  const y      = [];
  const postIds = [];
  for (const post of labelledPosts) {
    const combined = `${post.text || ""} ${post.headline || ""}`.trim();
    const tfidf    = tfidfVector(combined, vocab, idf, nTfidf);
    const numRaw   = buildNumFeatures(post);
    Xraw.push({ tfidf, numRaw });
    y.push(post.manualScore);
    postIds.push(post.postId || "");
  }

  // Scaler fitté sur l'ensemble complet (utilisé pour les deux passes)
  const scaler = fitScaler(Xraw.map(x => x.numRaw));
  const Xfull  = Xraw.map(x => [...x.tfidf, ...applyScaler(x.numRaw, scaler)]);

  // ── Passe 1 : split déterministe basé sur hash(postId) % 5
  // Les posts avec hash % 5 === 0 vont en validation (~20%), les autres en train.
  // Stable : ajouter des posts ne change pas le split des posts existants.
  const trainIdx = [], valIdx = [];
  for (let i = 0; i < Xfull.length; i++) {
    (hashPostId(postIds[i]) % 5 === 0 ? valIdx : trainIdx).push(i);
  }
  // Fallback : si val est vide (petit dataset avec mauvaise chance), prendre 1 sur 5
  if (valIdx.length === 0) trainIdx.forEach((_, k) => { if (k % 5 === 0) valIdx.push(trainIdx.splice(k, 1)[0]); });

  const Xtrain = trainIdx.map(i => Xfull[i]);
  const ytrain = trainIdx.map(i => y[i]);
  const Xval   = valIdx.map(i => Xfull[i]);
  const yval   = valIdx.map(i => y[i]);

  onProgress(1, 2);  // 50%
  const alpha = adaptiveAlpha(y.length);
  const { coef: coefVal, intercept: intVal } = solveRidge(Xtrain, ytrain, alpha);
  const maeVal = computeMAE(Xval, yval, coefVal, intVal);

  // ── Passe 2 : retrain sur 100% — c'est ce modèle qui est utilisé ──
  onProgress(2, 2);  // 100%
  const { coef, intercept } = solveRidge(Xfull, y, alpha);

  return {
    vocabulary:        vocab,
    idf:               Array.from(idf),  // recalculé sur tes données
    vocab_list:        vocabData.vocab_list,
    sublinear_tf:      true,
    scaler_mean:       Array.from(scaler.mean),
    scaler_scale:      Array.from(scaler.scale),
    ridge_coef:        coef,
    ridge_intercept:   intercept,
    n_tfidf_features:  nTfidf,
    n_num_features:    nNum,
    num_feature_names: vocabData.num_feature_names,
    trained_at:        new Date().toISOString(),
    n_samples:         y.length,
    alpha_used:        alpha,
    mae_val:           maeVal,   // MAE honnête (validation 20%)
  };
}

// ============================================================
// POPUP — initialisation
// ============================================================

chrome.storage.local.get(STORAGE_KEYS, async (result) => {
  const dataset       = result.bullshit_dataset || {};
  const mode          = result.bsd_mode || "filter";
  const threshold     = result.bsd_threshold ?? 7;
  const hideSponsored = result.bsd_hide_sponsored ?? true;
  const silentHide    = result.bsd_silent_hide ?? false;
  const customModel   = result.bsd_custom_model || null;

  // ── Mode toggle ──
  const btnFilter        = document.getElementById("btn-mode-filter");
  const btnCollect       = document.getElementById("btn-mode-collect");
  const thresholdSection = document.getElementById("threshold-section");

  function applyMode(m) {
    btnFilter.classList.toggle("active",  m === "filter");
    btnCollect.classList.toggle("active", m === "collect");
    thresholdSection.style.display = m === "filter" ? "block" : "none";
  }
  applyMode(mode);

  btnFilter.addEventListener("click", () => {
    chrome.storage.local.set({ bsd_mode: "filter" });
    applyMode("filter");
    notifyContentScript({ type: "BSD_MODE_CHANGED", mode: "filter", threshold: getCurrentThreshold() });
  });
  btnCollect.addEventListener("click", () => {
    chrome.storage.local.set({ bsd_mode: "collect" });
    applyMode("collect");
    notifyContentScript({ type: "BSD_MODE_CHANGED", mode: "collect", threshold: getCurrentThreshold() });
  });

  // ── Threshold slider ──
  const slider    = document.getElementById("threshold-slider");
  const threshVal = document.getElementById("threshold-val");
  slider.value = threshold;
  threshVal.textContent = threshold + "/10";
  slider.addEventListener("input", () => {
    const v = parseInt(slider.value);
    threshVal.textContent = v + "/10";
    chrome.storage.local.set({ bsd_threshold: v });
    notifyContentScript({ type: "BSD_THRESHOLD_CHANGED", threshold: v });
  });
  function getCurrentThreshold() { return parseInt(slider.value); }

  // ── Toggles sponsorisés / silencieux ──
  const toggleSponsored = document.getElementById("toggle-sponsored");
  toggleSponsored.checked = hideSponsored;
  toggleSponsored.addEventListener("change", () => {
    const v = toggleSponsored.checked;
    chrome.storage.local.set({ bsd_hide_sponsored: v });
    notifyContentScript({ type: "BSD_SPONSORED_CHANGED", hideSponsored: v });
  });

  const toggleSilent = document.getElementById("toggle-silent");
  toggleSilent.checked = silentHide;
  toggleSilent.addEventListener("change", () => {
    const v = toggleSilent.checked;
    chrome.storage.local.set({ bsd_silent_hide: v });
    notifyContentScript({ type: "BSD_SILENT_CHANGED", silent: v });
  });

  // ── Stats ──
  const posts = Object.values(dataset);

  if (posts.length === 0) {
    document.getElementById("empty-state").style.display  = "block";
    document.getElementById("loaded-state").style.display = "none";
    return;
  }

  document.getElementById("empty-state").style.display  = "none";
  document.getElementById("loaded-state").style.display = "block";

  const scores = posts.map(p => p.manualScore).filter(s => s !== undefined && s !== null);
  const avg    = scores.length > 0
    ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
    : "—";
  const high = scores.filter(s => s >= 7).length;

  document.getElementById("stat-total").textContent = posts.length;
  document.getElementById("stat-avg").textContent   = avg;
  document.getElementById("stat-high").textContent  = high;

  const dates = posts.map(p => p.savedAt).filter(Boolean).sort();
  if (dates.length > 0) {
    const last = new Date(dates[dates.length - 1]);
    document.getElementById("last-saved").textContent =
      "Dernière sauvegarde : " + last.toLocaleDateString("fr-FR") + " " +
      last.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  }

  // ── Export ──
  document.getElementById("btn-export").addEventListener("click", () => {
    const json = JSON.stringify(posts, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `bullshit_dataset_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ── Clear ──
  document.getElementById("btn-clear").addEventListener("click", () => {
    if (confirm(`Effacer les ${posts.length} posts sauvegardés ? Cette action est irréversible.`)) {
      chrome.storage.local.remove(["bullshit_dataset"], () => window.close());
    }
  });

  // ── Section réentraînement ──
  const btnTrain      = document.getElementById("btn-train");
  const btnReset      = document.getElementById("btn-reset-model");
  const trainBadge    = document.getElementById("train-badge");
  const trainDesc     = document.getElementById("train-desc");
  const trainProgress = document.getElementById("train-progress");
  const progressBar   = document.getElementById("progress-bar");
  const progressLbl   = document.getElementById("progress-label");
  const trainResult   = document.getElementById("train-result");
  const trainMAE      = document.getElementById("train-mae");
  const trainTS       = document.getElementById("train-ts");

  function showCustomModelState(model) {
    const mae = (model.mae_val ?? model.mae_train ?? 0).toFixed(2);
    const ts  = new Date(model.trained_at);
    trainBadge.textContent = "Personnalisé ✓";
    trainBadge.classList.add("active");
    trainDesc.textContent = `${model.n_samples} posts · MAE ${mae} pts`;
    trainMAE.textContent  = mae;
    trainTS.textContent   = ts.toLocaleDateString("fr-FR") + " " +
      ts.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    trainResult.classList.add("visible");
    btnReset.style.display = "block";
  }

  function showBaseModelState() {
    trainBadge.textContent = "Base";
    trainBadge.classList.remove("active");
    trainDesc.textContent = "Réentraîne Ridge sur tes labels.";
    trainResult.classList.remove("visible");
    btnReset.style.display = "none";
  }

  // Afficher l'état initial
  if (customModel) {
    showCustomModelState(customModel);
  } else {
    showBaseModelState();
  }

  // Désactiver si pas assez de posts labellisés
  const labelledCount = scores.length;
  if (labelledCount < MIN_POSTS_TO_TRAIN) {
    btnTrain.disabled = true;
    btnTrain.textContent = `🔁 Réentraîner (${labelledCount}/${MIN_POSTS_TO_TRAIN} min.)`;
  }

  // ── Clic Réentraîner ──
  btnTrain.addEventListener("click", async () => {
    if (labelledCount < MIN_POSTS_TO_TRAIN) return;

    if (labelledCount < 50) {
      const ok = confirm(
        `⚠️ Seulement ${labelledCount} posts labellisés.\n\n` +
        `Entraîner un modèle sur aussi peu de données est très déconseillé : ` +
        `les résultats seront peu fiables et très sensibles à chaque label individuel.\n\n` +
        `Il est recommandé d'atteindre au moins 50 posts avant d'entraîner.\n\n` +
        `Continuer quand même ?`
      );
      if (!ok) return;
    }

    btnTrain.disabled = true;
    btnReset.style.display = "none";
    btnTrain.textContent = "⏳ Validation…";
    trainProgress.classList.add("visible");
    trainResult.classList.remove("visible");
    progressBar.style.width = "0%";
    progressLbl.textContent = "Validation…";

    let vocabData;
    try {
      vocabData = await loadVocab();
    } catch (e) {
      btnTrain.disabled = false;
      btnTrain.textContent = "🔁 Réentraîner le modèle";
      alert("Erreur : impossible de charger tfidf_vocab.json");
      return;
    }

    setTimeout(async () => {
      const onProgress = (step, total) => {
        const pct = Math.round((step / total) * 100);
        progressBar.style.width = pct + "%";
        progressLbl.textContent = step === 1 ? "Entraînement…" : "Finalisation…";
        if (step === 1) btnTrain.textContent = "⏳ Entraînement…";
      };

      try {
        const newModel = await trainModel(posts, vocabData, onProgress);
        await chrome.storage.local.set({ bsd_custom_model: newModel });
        notifyContentScript({ type: "BSD_MODEL_UPDATED", model: newModel });

        progressBar.style.width = "100%";
        progressLbl.textContent = "Terminé";

        setTimeout(() => {
          trainProgress.classList.remove("visible");
          btnTrain.disabled = false;
          btnTrain.textContent = "🔁 Réentraîner le modèle";
          showCustomModelState(newModel);
        }, 800);

      } catch (err) {
        console.error("[BSD] Erreur entraînement :", err);
        trainProgress.classList.remove("visible");
        btnTrain.disabled = false;
        btnTrain.textContent = "🔁 Réentraîner le modèle";
        alert("Erreur pendant l'entraînement : " + err.message);
      }
    }, 50);
  });

  // ── Clic Revenir au modèle de base ──
  btnReset.addEventListener("click", () => {
    chrome.storage.local.remove(["bsd_custom_model"], () => {
      notifyContentScript({ type: "BSD_MODEL_RESET" });
      showBaseModelState();
    });
  });
});

function notifyContentScript(msg) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, msg).catch(() => {});
    }
  });
}
