// popup.js — v0.5

// ============================================================
// i18n — application des chaînes statiques
// ============================================================

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t(el.getAttribute("data-i18n"));
  });
}

// ============================================================
// Lang toggle FR | EN
// ============================================================

function initLangToggle() {
  const btnFr = document.getElementById("btn-lang-fr");
  const btnEn = document.getElementById("btn-lang-en");

  function updateActive() {
    btnFr.classList.toggle("active", LANG === "fr");
    btnEn.classList.toggle("active", LANG === "en");
  }
  updateActive();

  function switchTo(lang) {
    if (lang === LANG) return;
    sessionStorage.setItem("bsd_lang", lang);
    chrome.storage.local.set({ bsd_lang: lang });
    notifyContentScript({ type: "BSD_LANG_CHANGED", lang });
    window.location.reload();
  }

  btnFr.addEventListener("click", () => switchTo("fr"));
  btnEn.addEventListener("click", () => switchTo("en"));
}

// Au tout premier chargement du popup (sessionStorage vide),
// on récupère la préférence depuis chrome.storage.local et on
// initialise sessionStorage, puis on recharge si nécessaire.
async function seedLangFromStorage() {
  const inSession = sessionStorage.getItem("bsd_lang");
  if (inSession) return;

  const stored = await new Promise(resolve =>
    chrome.storage.local.get(["bsd_lang"], resolve)
  );
  const saved = stored.bsd_lang;
  if (saved === "fr" || saved === "en") {
    sessionStorage.setItem("bsd_lang", saved);
    if (saved !== LANG) {
      window.location.reload();
      return;
    }
  }
  chrome.storage.local.set({ bsd_lang: LANG });
}

// ============================================================
// CONSTANTES
// ============================================================

const STORAGE_KEYS = [
  "bullshit_dataset", "bsd_mode", "bsd_threshold",
  "bsd_hide_sponsored", "bsd_silent_hide", "bsd_custom_model", "bsd_stats"
];
const MIN_POSTS_TO_TRAIN = 20;

function adaptiveAlpha(n) {
  return Math.max(20 / n, 0.01);
}

// ============================================================
// TF-IDF
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

function computeIDF(texts, vocab, nFeatures) {
  const n  = texts.length;
  const df = new Float64Array(nFeatures);
  for (const text of texts) {
    const tokens = tokenize(text);
    const seen   = new Set();
    for (const tok of tokens) {
      if (tok in vocab && !seen.has(vocab[tok])) { df[vocab[tok]]++; seen.add(vocab[tok]); }
    }
    for (let i = 0; i < tokens.length - 1; i++) {
      const bigram = tokens[i] + " " + tokens[i + 1];
      if (bigram in vocab && !seen.has(vocab[bigram])) { df[vocab[bigram]]++; seen.add(vocab[bigram]); }
    }
  }
  const idf = new Float64Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) idf[i] = Math.log((1 + n) / (1 + df[i])) + 1;
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
  const text     = post.text || "";
  const headline = post.headline || "";
  const wordCount = text.split(/\s+/).length;
  const emojiCount = [...text].filter(c => c.codePointAt(0) > 0x1F000).length;
  return [
    parseCount(post.likes), parseCount(post.comments),
    text.length, wordCount, fcFlag(post.feedContext),
    emojiCount / Math.max(wordCount, 1), headline.length,
  ];
}

// ============================================================
// StandardScaler
// ============================================================

function fitScaler(matrix) {
  const n = matrix[0].length;
  const mean  = new Float64Array(n);
  const scale = new Float64Array(n);
  for (const row of matrix) for (let j = 0; j < n; j++) mean[j] += row[j];
  for (let j = 0; j < n; j++) mean[j] /= matrix.length;
  for (const row of matrix) for (let j = 0; j < n; j++) scale[j] += (row[j] - mean[j]) ** 2;
  const ddof = Math.max(matrix.length - 1, 1);
  for (let j = 0; j < n; j++) scale[j] = Math.sqrt(scale[j] / ddof) || 1;
  return { mean, scale };
}

function applyScaler(row, scaler) {
  return row.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]);
}

// ============================================================
// Ridge
// ============================================================

function solveLinear(A, b, n) {
  const M = new Float64Array(A);
  const x = new Float64Array(b);
  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(M[col * n + col]), maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(M[row * n + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
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
  const w = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= M[i * n + j] * w[j];
    w[i] = s / M[i * n + i];
  }
  return w;
}

function solveRidge(X, y, alpha) {
  const m = X.length, n = X[0].length;
  const xMean = new Float64Array(n);
  let yMean = 0;
  for (let i = 0; i < m; i++) { for (let j = 0; j < n; j++) xMean[j] += X[i][j]; yMean += y[i]; }
  for (let j = 0; j < n; j++) xMean[j] /= m;
  yMean /= m;
  const Xc = X.map(row => row.map((v, j) => v - xMean[j]));
  const yc  = y.map(v => v - yMean);
  const XtX = new Float64Array(n * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) XtX[j * n + k] += Xc[i][j] * Xc[i][k];
  for (let j = 0; j < n; j++) XtX[j * n + j] += alpha;
  const Xty = new Float64Array(n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Xty[j] += Xc[i][j] * yc[i];
  const w = solveLinear(XtX, Xty, n);
  let intercept = yMean;
  for (let j = 0; j < n; j++) intercept -= xMean[j] * w[j];
  return { coef: Array.from(w), intercept };
}

function computeMAE(X, y, coef, intercept) {
  let mae = 0;
  for (let i = 0; i < X.length; i++) {
    let pred = intercept;
    for (let j = 0; j < coef.length; j++) pred += coef[j] * X[i][j];
    mae += Math.abs(Math.max(0, Math.min(10, pred)) - y[i]);
  }
  return mae / X.length;
}

function hashPostId(id) {
  let h = 5381;
  for (let i = 0; i < (id || "").length; i++) { h = ((h << 5) + h) ^ (id || "").charCodeAt(i); h = h | 0; }
  return Math.abs(h);
}

// ============================================================
// Pipeline d'entraînement
// ============================================================

async function trainModel(posts, vocabData, onProgress) {
  const vocab  = vocabData.vocabulary;
  const nTfidf = vocabData.n_tfidf_features;
  const nNum   = vocabData.n_num_features;
  const labelledPosts = posts.filter(p => p.manualScore !== null && p.manualScore !== undefined);
  const corpus = labelledPosts.map(p => `${p.text || ""} ${p.headline || ""}`.trim());
  const idf    = computeIDF(corpus, vocab, nTfidf);
  const Xraw = [], y = [], postIds = [];
  for (const post of labelledPosts) {
    const combined = `${post.text || ""} ${post.headline || ""}`.trim();
    Xraw.push({ tfidf: tfidfVector(combined, vocab, idf, nTfidf), numRaw: buildNumFeatures(post) });
    y.push(post.manualScore);
    postIds.push(post.postId || "");
  }
  const scaler = fitScaler(Xraw.map(x => x.numRaw));
  const Xfull  = Xraw.map(x => [...x.tfidf, ...applyScaler(x.numRaw, scaler)]);
  const trainIdx = [], valIdx = [];
  for (let i = 0; i < Xfull.length; i++) (hashPostId(postIds[i]) % 5 === 0 ? valIdx : trainIdx).push(i);
  if (valIdx.length === 0) trainIdx.forEach((_, k) => { if (k % 5 === 0) valIdx.push(trainIdx.splice(k, 1)[0]); });
  onProgress(1, 2);
  const alpha = adaptiveAlpha(y.length);
  const { coef: coefVal, intercept: intVal } = solveRidge(trainIdx.map(i => Xfull[i]), trainIdx.map(i => y[i]), alpha);
  const maeVal = computeMAE(valIdx.map(i => Xfull[i]), valIdx.map(i => y[i]), coefVal, intVal);
  onProgress(2, 2);
  const { coef, intercept } = solveRidge(Xfull, y, alpha);
  return {
    vocabulary: vocab, idf: Array.from(idf), vocab_list: vocabData.vocab_list,
    sublinear_tf: true, scaler_mean: Array.from(scaler.mean), scaler_scale: Array.from(scaler.scale),
    ridge_coef: coef, ridge_intercept: intercept,
    n_tfidf_features: nTfidf, n_num_features: nNum, num_feature_names: vocabData.num_feature_names,
    trained_at: new Date().toISOString(), n_samples: y.length, alpha_used: alpha, mae_val: maeVal,
  };
}

// ============================================================
// QUALITÉ MODÈLE — affichage dans la stat card
// ============================================================

// Formule de pondération (même que content.js)
function computeW(maeCustom, maeBase, n) {
  const nScale = 20, k = 4;
  const wN   = 1 - Math.exp(-n / nScale);
  const wMae = 0.1 + 0.9 / (1 + Math.exp(-k * (maeBase - maeCustom)));
  return wN * wMae;
}

function personalizationLabel(w) {
  if (w === null || w === undefined) return { text: t("model_quality_none"),       cls: "quality-none"      };
  if (w < 0.10)                      return { text: t("model_quality_very_low"),   cls: "quality-none"      };
  if (w < 0.30)                      return { text: t("model_quality_low"),        cls: "quality-average"   };
  if (w < 0.50)                      return { text: t("model_quality_medium"),     cls: "quality-good"      };
  if (w < 0.75)                      return { text: t("model_quality_high"),       cls: "quality-very-good" };
  if (w < 0.90)                      return { text: t("model_quality_very_high"),  cls: "quality-excellent" };
  return                              { text: t("model_quality_optimal"),           cls: "quality-excellent" };
}

function updatePersonalizationStat(w, noData = false) {
  const el = document.getElementById("stat-model-quality");
  if (!el) return;
  el.className = "val";
  if (noData) {
    el.textContent = t("model_quality_no_data");
    el.classList.add("quality-no-data");
    return;
  }
  const { text, cls } = personalizationLabel(w);
  el.textContent = text;
  el.classList.add(cls);
}

// ============================================================
// POPUP — initialisation
// ============================================================

applyI18n();
initLangToggle();

seedLangFromStorage().then(() => {});

chrome.storage.local.get(STORAGE_KEYS, async (result) => {
  const dataset       = result.bullshit_dataset || {};
  const mode          = result.bsd_mode || "filter";
  const threshold     = result.bsd_threshold ?? 7;
  const hideSponsored = result.bsd_hide_sponsored ?? true;
  const silentHide    = result.bsd_silent_hide ?? false;
  const customModel   = result.bsd_custom_model || null;

  // ── Stats filtre ──
  const stats   = result.bsd_stats || {};
  const today   = (function() {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
  })();
  const isToday = stats.today_date === today;
  const todayBs        = isToday ? (stats.today_bs        || 0) : 0;
  const todaySponsored = isToday ? (stats.today_sponsored || 0) : 0;
  const totalHidden    = (stats.total_bs || 0) + (stats.total_sponsored || 0);
  const scrollRawM = (stats.total_scroll_px || 0) / (96 * 39.37);
  const scrollStr  = scrollRawM >= 1000
    ? (scrollRawM / 1000).toFixed(2) + " km"
    : scrollRawM.toFixed(1) + " m";

  const elTodayBs        = document.getElementById("fstat-today-bs");
  const elTodaySponsored = document.getElementById("fstat-today-sponsored");
  const elTotal          = document.getElementById("fstat-total");
  const elScroll         = document.getElementById("fstat-scroll");
  if (elTodayBs)        elTodayBs.textContent        = todayBs;
  if (elTodaySponsored) elTodaySponsored.textContent  = todaySponsored;
  if (elTotal)          elTotal.textContent           = totalHidden;
  if (elScroll)         elScroll.textContent          = scrollStr;

  const btnFilter        = document.getElementById("btn-mode-filter");
  const btnCollect       = document.getElementById("btn-mode-collect");
  const thresholdSection = document.getElementById("threshold-section");

  function applyMode(m) {
    btnFilter.classList.toggle("active",  m === "filter");
    btnCollect.classList.toggle("active", m === "collect");
    thresholdSection.style.display = m === "filter" ? "block" : "none";
    const filterStatsEl = document.getElementById("filter-stats");
    if (filterStatsEl) filterStatsEl.style.display = m === "filter" ? "block" : "none";
    // La grille collecte ne s'affiche qu'en mode collecte
    const loadedEl = document.getElementById("loaded-state");
    if (loadedEl) loadedEl.style.display = m === "collect" ? "" : "none";
    const actionsEl = document.getElementById("actions-secondary");
    if (actionsEl) actionsEl.style.display = m === "collect" ? "flex" : "none";
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

  // Sync toggle if silent was enabled from a placeholder button
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.bsd_silent_hide) {
      toggleSilent.checked = changes.bsd_silent_hide.newValue;
    }
  });

  const posts = Object.values(dataset);

  if (posts.length === 0) {
    document.getElementById("empty-state").style.display  = "block";
    document.getElementById("loaded-state").style.display = "none";
    // Afficher quand même le message dans la stat card si elle était visible
    // → rien à faire ici, loaded-state est caché
    return;
  }

  document.getElementById("empty-state").style.display  = "none";
  // loaded-state visible seulement en mode collecte (applyMode le gère aussi)
  document.getElementById("loaded-state").style.display = mode === "collect" ? "block" : "none";

  const scores = posts.map(p => p.manualScore).filter(s => s !== undefined && s !== null);
  const avg    = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : "—";
  const high   = scores.filter(s => s >= 7).length;

  document.getElementById("stat-total").textContent = posts.length;
  document.getElementById("stat-avg").textContent   = avg;
  document.getElementById("stat-high").textContent  = high;

  // Qualité du modèle — sera calculée par showBaseModelState() ou showCustomModelState()
  // appelés juste après la définition de ces fonctions ci-dessous

  const locale = LANG === "fr" ? "fr-FR" : "en-GB";

  document.getElementById("btn-import").addEventListener("click", () => {
    document.getElementById("import-file").click();
  });

  document.getElementById("import-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const raw = JSON.parse(ev.target.result);
        // Accepte tableau ou objet {postId: post}
        const incoming = Array.isArray(raw) ? raw : Object.values(raw);
        if (!incoming.length || typeof incoming[0] !== "object") throw new Error();
        if (!confirm(t("popup_import_confirm", incoming.length, posts.length))) return;
        chrome.storage.local.get(["bullshit_dataset"], (result) => {
          const dataset = result.bullshit_dataset || {};
          let added = 0;
          for (const post of incoming) {
            const id = post.postId || post.id;
            if (!id) continue;
            if (!dataset[id]) { dataset[id] = post; added++; }
          }
          chrome.storage.local.set({ bullshit_dataset: dataset }, () => {
            window.location.reload();
          });
        });
      } catch {
        alert(t("popup_import_error"));
      }
    };
    reader.readAsText(file);
    e.target.value = ""; // reset pour permettre re-import du même fichier
  });

  document.getElementById("btn-export").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(posts, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = `bullshit_dataset_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("btn-clear").addEventListener("click", () => {
    if (confirm(t("popup_clear_confirm", posts.length))) {
      chrome.storage.local.remove(["bullshit_dataset"], () => window.close());
    }
  });

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

  function computeAndShowBaseMae() {
    const labelledPosts = posts.filter(p => p.manualScore !== null && p.manualScore !== undefined);
    if (labelledPosts.length === 0) { updatePersonalizationStat(null); return; }
    loadVocab().then(vocabData => {
      const vocab       = vocabData.vocabulary;
      const nTfidf      = vocabData.n_tfidf_features;
      const idf         = vocabData.idf;
      const coef        = vocabData.ridge_coef;
      const intercept   = vocabData.ridge_intercept;
      const scalerMean  = vocabData.scaler_mean;
      const scalerScale = vocabData.scaler_scale;
      let maeSum = 0;
      for (const post of labelledPosts) {
        const combined  = `${post.text || ""} ${post.headline || ""}`.trim();
        const tfidfVec  = tfidfVector(combined, vocab, idf, nTfidf);
        const numRaw    = buildNumFeatures(post);
        const numScaled = numRaw.map((v, i) => (v - scalerMean[i]) / scalerScale[i]);
        let pred = intercept;
        for (let i = 0; i < nTfidf; i++) pred += tfidfVec[i] * coef[i];
        for (let i = 0; i < numScaled.length; i++) pred += numScaled[i] * coef[nTfidf + i];
        pred = Math.max(0, Math.min(10, pred));
        maeSum += Math.abs(pred - post.manualScore);
      }
      const maeBase = maeSum / labelledPosts.length;
      chrome.storage.local.set({ bsd_mae_base: maeBase });
      updatePersonalizationStat(null); // pas de modèle custom → personnalisation nulle
    }).catch(e => {
      console.error("[BSD] Base MAE computation failed:", e);
      updatePersonalizationStat(null);
    });
  }

  function showCustomModelState(model) {
    const mae = (model.mae_val ?? model.mae_train ?? 0).toFixed(2);
    const ts  = new Date(model.trained_at);
    const tsStr = ts.toLocaleDateString(locale) + " " +
      ts.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" });
    trainBadge.textContent = t("train_badge_custom");
    trainBadge.classList.add("active");
    trainDesc.textContent = t("train_desc_custom", model.n_samples);
    trainMAE.textContent  = mae;
    trainTS.textContent   = tsStr;
    // Ligne fusionnée : MAE · date
    trainResult.innerHTML = `MAE : <span class="mae">${mae} pts</span> &nbsp;·&nbsp; <span class="ts">${tsStr}</span>`;
    trainResult.classList.add("visible");
    btnReset.style.display = "block";
    chrome.storage.local.get(["bsd_mae_base"], (r) => {
      const maeCustom = model.mae_val ?? model.mae_train ?? null;
      const maeBase   = r.bsd_mae_base ?? null;
      const n         = model.n_samples ?? 0;
      if (maeCustom !== null && maeBase !== null) {
        updatePersonalizationStat(computeW(maeCustom, maeBase, n));
      } else {
        updatePersonalizationStat(null);
      }
    });
  }

  function showBaseModelState() {
    trainBadge.textContent = t("train_badge_base");
    trainBadge.classList.remove("active");
    trainDesc.textContent = t("train_desc_base");
    trainResult.classList.remove("visible");
    btnReset.style.display = "none";
    updatePersonalizationStat(null); // pas de modèle custom → personnalisation nulle
    computeAndShowBaseMae(); // recalcule et stocke mae_base
  }

  btnTrain.textContent = t("train_btn");
  if (customModel) showCustomModelState(customModel);
  else showBaseModelState();

  const labelledCount = scores.length;
  if (labelledCount < MIN_POSTS_TO_TRAIN) {
    btnTrain.disabled = true;
    btnTrain.textContent = t("train_btn_min", labelledCount, MIN_POSTS_TO_TRAIN);
  }

  btnTrain.addEventListener("click", async () => {
    if (labelledCount < MIN_POSTS_TO_TRAIN) return;

    btnTrain.disabled = true;
    btnReset.style.display = "none";
    btnTrain.textContent = t("train_btn_validating");
    trainProgress.classList.add("visible");
    trainResult.classList.remove("visible");
    progressBar.style.width = "0%";
    progressLbl.textContent = t("train_progress_val");

    let vocabData;
    try {
      vocabData = await loadVocab();
    } catch (e) {
      btnTrain.disabled = false;
      btnTrain.textContent = t("train_btn");
      alert(t("train_error_vocab"));
      return;
    }

    setTimeout(async () => {
      const onProgress = (step, total) => {
        progressBar.style.width = Math.round((step / total) * 100) + "%";
        progressLbl.textContent = step === 1 ? t("train_progress_val") : t("train_progress_final");
        if (step === 1) btnTrain.textContent = t("train_btn_training");
      };
      try {
        const newModel = await trainModel(posts, vocabData, onProgress);
        await chrome.storage.local.set({ bsd_custom_model: newModel });
        notifyContentScript({ type: "BSD_MODEL_UPDATED", model: newModel });
        progressBar.style.width = "100%";
        progressLbl.textContent = t("train_progress_done");
        setTimeout(() => {
          trainProgress.classList.remove("visible");
          btnTrain.disabled = false;
          btnTrain.textContent = t("train_btn");
          showCustomModelState(newModel);
        }, 800);
      } catch (err) {
        console.error("[BSD] Training error:", err);
        trainProgress.classList.remove("visible");
        btnTrain.disabled = false;
        btnTrain.textContent = t("train_btn");
        alert(t("train_error_generic", err.message));
      }
    }, 50);
  });

  btnReset.addEventListener("click", () => {
    chrome.storage.local.remove(["bsd_custom_model"], () => {
      notifyContentScript({ type: "BSD_MODEL_RESET" });
      showBaseModelState();
    });
  });
});

function notifyContentScript(msg) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id) chrome.tabs.sendMessage(tabs[0].id, msg).catch(() => {});
  });
}
