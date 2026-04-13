// ============================================================
// LinkedIn Bullshit Detector — content.js (v0.5)
// ============================================================

const DEBUG = true;
const log = (...args) => DEBUG && console.log("[BSD]", ...args);

// ── État global ──
let MODE           = "filter";
let THRESHOLD      = 7;
let HIDE_SPONSORED = true;
let SILENT_HIDE    = false;
let BASE_MODEL     = null;   // toujours tfidf_vocab.json
let CUSTOM_MODEL   = null;   // modèle entraîné par l'utilisateur (ou null)
let MAE_BASE       = null;   // stockée par le popup
let N_LABELLED     = 0;      // nombre de posts labellisés

// ============================================================
// MODÈLE
// ============================================================

function computeW(maeCustom, maeBase, n) {
  const nScale = 20, k = 4;
  const wN   = 1 - Math.exp(-n / nScale);
  const wMae = 0.1 + 0.9 / (1 + Math.exp(-k * (maeBase - maeCustom)));
  return wN * wMae;
}

async function loadModel() {
  const stored = await chrome.storage.local.get([
    "bsd_custom_model", "bsd_mae_base", "bullshit_dataset"
  ]);

  // Toujours charger le modèle de base
  const url = chrome.runtime.getURL("tfidf_vocab.json");
  const res = await fetch(url);
  BASE_MODEL = await res.json();

  CUSTOM_MODEL = stored.bsd_custom_model || null;
  MAE_BASE     = stored.bsd_mae_base     || null;
  N_LABELLED   = Object.values(stored.bullshit_dataset || {})
    .filter(p => p.manualScore !== null && p.manualScore !== undefined).length;

  if (CUSTOM_MODEL) {
    log("🧠 Custom model loaded —", CUSTOM_MODEL.n_samples, "posts · MAE", CUSTOM_MODEL.mae_val?.toFixed(2));
  }
  log("🧠 Base model loaded —", BASE_MODEL.n_tfidf_features, "TF-IDF features");
  log("🧠 MAE base =", MAE_BASE, "· n_labelled =", N_LABELLED);
}

function tokenize(text) {
  return (text || "").toLowerCase().match(/[a-z0-9\u00c0-\u017e]+/g) || [];
}

function tfidfVectorFromModel(text, model) {
  const tokens = tokenize(text);
  const vocab  = model.vocabulary;
  const idf    = model.idf;
  const n      = model.n_tfidf_features;
  const tf     = new Float64Array(n);
  for (const tok of tokens) { if (tok in vocab) tf[vocab[tok]] += 1; }
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = tokens[i] + " " + tokens[i + 1];
    if (bigram in vocab) tf[vocab[bigram]] += 1;
  }
  for (let i = 0; i < n; i++) { if (tf[i] > 0) tf[i] = 1 + Math.log(tf[i]); }
  for (let i = 0; i < n; i++) tf[i] *= idf[i];
  let norm = 0;
  for (let i = 0; i < n; i++) norm += tf[i] * tf[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < n; i++) tf[i] /= norm;
  return tf;
}

function scoreFromModel(postData, model) {
  const combined  = `${postData.text || ""} ${postData.headline || ""}`.trim();
  const tfidfVec  = tfidfVectorFromModel(combined, model);
  const text      = postData.text || "";
  const textWords = text.split(/\s+/);
  const wordCount = textWords.length;
  const emojiCount = [...text].filter(c => c.codePointAt(0) > 0x1F000).length;

  // Features de base (6)
  const baseNum = [
    parseCount(postData.likes), parseCount(postData.comments),
    text.length, wordCount,
    emojiCount / Math.max(wordCount, 1), (postData.headline || "").length,
  ];

  // Features phrases courtes (4)
  const shortSent = [3, 5, 7, 10].map(k => countShortSentences(text, k));

  // Features top-15 emojis — lues depuis model.top_emojis (null = base model)
  const topEmojis = model.top_emojis || null;
  const emojiFeats = topEmojis
    ? topEmojis.map(em => [...text].filter(c => c === em).length)
    : new Array(10).fill(0);

  const rawNum = [...baseNum, ...shortSent, ...emojiFeats];
  const nNum   = model.n_num_features;
  const rawNumTrunc = rawNum.slice(0, nNum);

  const scaledNum = rawNumTrunc.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i]);
  const coef = model.ridge_coef;
  let score  = model.ridge_intercept;
  for (let i = 0; i < model.n_tfidf_features; i++) score += tfidfVec[i] * coef[i];
  for (let i = 0; i < nNum; i++) score += scaledNum[i] * coef[model.n_tfidf_features + i];

  if (model.use_sigmoid) {
    const k = model.sigmoid_slope ?? 1.0;
    return Math.max(0, Math.min(10, 10 / (1 + Math.exp(-k * score))));
  }
  return Math.max(0, Math.min(10, score));
}

function parseCount(s) {
  const digits = String(s || "").replace(/[^\d]/g, "");
  return digits ? parseInt(digits, 10) : 0;
}

function feedContextFlag(fc) {
  return fc && fc.toLowerCase().includes("profil") ? 1 : 0;
}

function countShortSentences(text, maxWords) {
  if (!text) return 0;
  const sentences = text.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 0);
  return sentences.filter(s => s.split(/\s+/).length < maxWords).length;
}

function computeModelScore(postData) {
  if (!BASE_MODEL) return { score: 0 };

  const scoreBase = scoreFromModel(postData, BASE_MODEL);

  if (!CUSTOM_MODEL || MAE_BASE === null) {
    return { score: Math.round(scoreBase) };
  }

  const scoreCustom = scoreFromModel(postData, CUSTOM_MODEL);
  const maeCustom   = CUSTOM_MODEL.mae_val ?? CUSTOM_MODEL.mae_train ?? MAE_BASE;
  const w           = computeW(maeCustom, MAE_BASE, N_LABELLED);
  const blended     = w * scoreCustom + (1 - w) * scoreBase;

  return { score: Math.round(blended) };
}

// ============================================================
// DÉTECTION — posts sponsorisés
// ============================================================

function isSponsored(postEl) {
  return [...postEl.querySelectorAll('p[componentkey], span[componentkey]')]
    .some(el => el.textContent.trim() === 'Sponsorisé' || el.textContent.trim() === 'Promoted');
}

// ============================================================
// EXTRACTION
// ============================================================

function extractPostData(postEl) {
  const leaves = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0);
  const texts = leaves.map(el => el.innerText.trim());

  const NOISE = ["post du fil d'actualité", "en fonction de votre profil et de votre activité",
    "j'aime", "commenter", "republier", "envoyer", "suivre", "signaler", "…", "plus",
    "like", "comment", "repost", "send", "follow", "report", "more"];

  let author = "";
  const menuBtn = postEl.querySelector(
    'button[aria-label^="Ouvrir le menu de commandes pour le post de"], button[aria-label^="Open control menu for post by"]'
  );
  if (menuBtn) {
    author = menuBtn.getAttribute("aria-label")
      .replace("Ouvrir le menu de commandes pour le post de", "")
      .replace("Open control menu for post by", "")
      .trim();
  }

  let headline = "";
  const authorIdx = texts.indexOf(author);
  if (authorIdx !== -1) {
    for (let i = authorIdx + 1; i < texts.length; i++) {
      const tx = texts[i];
      if (/^•\s*\d/.test(tx)) continue;
      if (NOISE.includes(tx.toLowerCase())) continue;
      headline = tx;
      break;
    }
  }

  let text = "", maxLen = 80;
  postEl.querySelectorAll('p').forEach(el => {
    if (el.children.length > 5) return;
    const tx = el.innerText.trim();
    if (NOISE.includes(tx.toLowerCase())) return;
    if (tx.length > maxLen) { maxLen = tx.length; text = tx; }
  });

  const likesRaw = texts.find(tx =>
    tx.toLowerCase().includes("réaction") || tx.toLowerCase().includes("reaction") || tx.toLowerCase().includes("like")
  );
  const likes = likesRaw ? likesRaw.split("\n")[0].trim() : "0";

  const commentsRaw = texts.find(tx =>
    tx.toLowerCase().includes("commentaire") || tx.toLowerCase().includes("comment")
  );
  const comments = commentsRaw ? commentsRaw.split("\n")[0].trim() : "0";

  const linkEl  = postEl.querySelector('a[href*="/posts/"], a[href*="activity"], a[href*="/feed/update/"]');
  const postUrl = linkEl ? linkEl.href.split("?")[0] : "";

  const allLeaves   = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0)
    .map(el => el.innerText.trim());
  const feedContext = (allLeaves[1] && allLeaves[1] !== author) ? allLeaves[1] : "";

  return { text, author, headline, likes, comments, postUrl, feedContext };
}

function generatePostId(postEl) {
  const key = postEl.getAttribute("componentkey") || "";
  if (key) return "post_" + key.slice(0, 20).replace(/[^a-z0-9]/gi, "");
  const str = postEl.innerText.slice(0, 100);
  let hash = 0;
  for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
  return "post_" + Math.abs(hash).toString(36);
}

// ============================================================
// SAUVEGARDE
// ============================================================

async function savePost(postId, data, manualScore, autoScore) {
  return new Promise((resolve) => {
    chrome.storage.local.get(["bullshit_dataset"], (result) => {
      const dataset = result.bullshit_dataset || {};
      const { author, ...dataWithoutAuthor } = data;
      dataset[postId] = {
        ...dataWithoutAuthor, postId, manualScore, autoScore,
        savedAt: new Date().toISOString(),
        pageUrl: window.location.href,
      };
      chrome.storage.local.set({ bullshit_dataset: dataset }, resolve);
    });
  });
}

async function getSavedScore(postId) {
  return new Promise((resolve) => {
    chrome.storage.local.get(["bullshit_dataset"], (result) => {
      const dataset = result.bullshit_dataset || {};
      const entry = dataset[postId];
      resolve(entry?.manualScore ?? null);
    });
  });
}

// ============================================================
// STATISTIQUES DE MASQUAGE
// ============================================================

function todayString() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

function incrementStats(sponsored, scrollPx) {
  chrome.storage.local.get(["bsd_stats"], (result) => {
    const today = todayString();
    const s = result.bsd_stats || {};
    if (s.today_date !== today) {
      s.today_date      = today;
      s.today_bs        = 0;
      s.today_sponsored = 0;
    }
    if (sponsored) {
      s.today_sponsored = (s.today_sponsored || 0) + 1;
      s.total_sponsored = (s.total_sponsored || 0) + 1;
    } else {
      s.today_bs    = (s.today_bs    || 0) + 1;
      s.total_bs    = (s.total_bs    || 0) + 1;
    }
    s.total_scroll_px = (s.total_scroll_px || 0) + scrollPx;
    chrome.storage.local.set({ bsd_stats: s });
  });
}

// ============================================================
// MODE FILTRE — placeholder cliquable
// ============================================================

function applyFilterMode(postEl, score, sponsored = false) {
  const scrollPx = postEl.offsetHeight || 0;

  postEl.querySelectorAll(':scope > *').forEach(el => {
    if (!el.classList.contains('bsd-placeholder')) el.style.display = "none";
  });
  if (postEl.querySelector('.bsd-placeholder')) return;

  // Incrémenter les stats même en mode silencieux
  incrementStats(sponsored, scrollPx);

  if (SILENT_HIDE) return;

  const label = sponsored ? t("content_sponsored") : t("content_hidden", score);

  const placeholder = document.createElement("div");
  placeholder.className = "bsd-placeholder" + (sponsored ? " bsd-placeholder-sponsored" : "");
  placeholder.innerHTML = `
    <span>${label}</span>
    <div class="bsd-placeholder-actions">
      <button class="bsd-show-btn">${t("content_show")}</button>
      <button class="bsd-silent-btn">${t("content_silent_btn")}</button>
    </div>
  `;

  placeholder.querySelector(".bsd-show-btn").addEventListener("click", () => {
    placeholder.remove();
    postEl.querySelectorAll(':scope > *').forEach(el => el.style.display = "");
    postEl.dataset.bsdDone = "shown";
  });

  placeholder.querySelector(".bsd-silent-btn").addEventListener("click", () => {
    // Activer le masquage silencieux globalement
    SILENT_HIDE = true;
    chrome.storage.local.set({ bsd_silent_hide: true });
    // Supprimer tous les placeholders existants (ils disparaissent silencieusement)
    document.querySelectorAll('.bsd-placeholder').forEach(p => p.remove());
    log("🔇 Silent mode enabled from placeholder");
  });

  postEl.appendChild(placeholder);
}

// ============================================================
// MODE COLLECTE — widget notation
// ============================================================

async function createCollectWidget(postEl, postId, postData, autoScore) {
  const isSuspect = autoScore >= 4;
  const autoLevel = autoScore >= 7 ? "🔴" : autoScore >= 4 ? "🟠" : "🟢";

  const isPromoted = postData.feedContext?.toLowerCase().includes("profil") ||
                     postData.feedContext?.toLowerCase().includes("activité") ||
                     postData.feedContext?.toLowerCase().includes("activity") ||
                     postData.feedContext?.toLowerCase().includes("profile");
  const promotedHtml = isPromoted
    ? `<div class="bsd-promoted">${t("widget_promoted", postData.feedContext)}</div>`
    : "";

  // Récupérer une note déjà sauvegardée si elle existe
  const savedScore = await getSavedScore(postId);
  const initialSliderVal = savedScore !== null ? savedScore : 5;
  const initialLabelVal  = savedScore !== null ? (savedScore + "/10 " + t("widget_saved")) : t("widget_manual_none");

  const widget = document.createElement("div");
  widget.className = "bsd-widget" + (isSuspect ? " bsd-suspect" : "");
  widget.dataset.postId = postId;
  widget.innerHTML = `
    ${promotedHtml}
    <div class="bsd-auto-badge">
      <span class="bsd-auto-label">${t("widget_auto_label")}</span>
      <span>${autoLevel} ${t("widget_model_score", autoScore)}</span>
    </div>
    <div class="bsd-slider-wrap">
      <label>${t("widget_manual_label")} <strong class="bsd-manual-val">${initialLabelVal}</strong></label>
      <div class="bsd-slider-row">
        <span class="bsd-tick bsd-tick-label">${t("widget_scale_low")}</span>
        <input type="range" class="bsd-slider" min="0" max="10" step="1" value="${initialSliderVal}">
        <span class="bsd-tick bsd-tick-label">${t("widget_scale_high")}</span>
      </div>
    </div>
  `;

  const slider    = widget.querySelector(".bsd-slider");
  const manualVal = widget.querySelector(".bsd-manual-val");

  // Mise à jour du label pendant le glissement
  slider.addEventListener("input", () => {
    manualVal.textContent = slider.value + "/10";
    manualVal.classList.remove("bsd-saved-label");
  });

  // Sauvegarde automatique au relâchement
  slider.addEventListener("change", async () => {
    await savePost(postId, postData, parseInt(slider.value), autoScore);
    manualVal.textContent = slider.value + "/10 " + t("widget_saved");
    manualVal.classList.add("bsd-saved-label");
  });

  return widget;
}

// ============================================================
// TRAITEMENT D'UN POST
// ============================================================

async function processPost(postEl) {
  if (postEl.dataset.bsdDone === "shown") return;
  if (postEl.dataset.bsdDone === "1") return;

  const menuBtn = postEl.querySelector(
    'button[aria-label^="Ouvrir le menu de commandes pour le post de"], button[aria-label^="Open control menu for post by"]'
  );
  if (!menuBtn) return;

  postEl.dataset.bsdDone = "1";

  if (HIDE_SPONSORED && isSponsored(postEl)) {
    log("📢 Sponsored hidden");
    applyFilterMode(postEl, 0, true);
    return;
  }

  const data             = extractPostData(postEl);
  const postId           = generatePostId(postEl);
  const { score } = computeModelScore(data);

  log("✅", data.author || "(no author)", `[${score}/10]`, "—", data.text.slice(0, 50));

  if (MODE === "filter" && score >= THRESHOLD) {
    applyFilterMode(postEl, score);
  } else if (MODE === "collect") {
    const widget = await createCollectWidget(postEl, postId, data, score);
    postEl.appendChild(widget);
  }
}

// ============================================================
// RESET
// ============================================================

function resetAllPosts() {
  document.querySelectorAll('[data-bsd-done]').forEach(el => {
    el.querySelectorAll('.bsd-widget, .bsd-placeholder').forEach(w => w.remove());
    el.querySelectorAll(':scope > *').forEach(c => c.style.display = "");
    delete el.dataset.bsdDone;
  });
  findAndProcessPosts();
}

// ============================================================
// MESSAGES
// ============================================================

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "BSD_MODE_CHANGED") {
    MODE = msg.mode;
    if (msg.threshold !== undefined) THRESHOLD = msg.threshold;
    resetAllPosts();
  }
  if (msg.type === "BSD_THRESHOLD_CHANGED") {
    THRESHOLD = msg.threshold;
    resetAllPosts();
  }
  if (msg.type === "BSD_SPONSORED_CHANGED") {
    HIDE_SPONSORED = msg.hideSponsored;
    resetAllPosts();
  }
  if (msg.type === "BSD_SILENT_CHANGED") {
    SILENT_HIDE = msg.silent;
    resetAllPosts();
  }
  if (msg.type === "BSD_MODEL_UPDATED") {
    CUSTOM_MODEL = msg.model;
    // Récupérer mae_base et n_labelled à jour
    chrome.storage.local.get(["bsd_mae_base", "bullshit_dataset"], (r) => {
      MAE_BASE   = r.bsd_mae_base || null;
      N_LABELLED = Object.values(r.bullshit_dataset || {})
        .filter(p => p.manualScore !== null && p.manualScore !== undefined).length;
      log("🧠 Model updated — w =", MAE_BASE ? computeW(
        CUSTOM_MODEL.mae_val ?? CUSTOM_MODEL.mae_train, MAE_BASE, N_LABELLED
      ).toFixed(2) : "?");
      resetAllPosts();
    });
  }
  if (msg.type === "BSD_MODEL_RESET") {
    CUSTOM_MODEL = null;
    MAE_BASE     = null;
    resetAllPosts();
  }
  // ── Changement de langue ──
  if (msg.type === "BSD_LANG_CHANGED") {
    LANG = msg.lang;
    log("🌐 Lang →", LANG);
    resetAllPosts();
  }
});

// ============================================================
// SCAN + OBSERVER
// ============================================================

function findAndProcessPosts() {
  const all   = document.querySelectorAll('[componentkey^="expanded"]');
  const posts = [...all].filter(el => !el.parentElement?.closest('[componentkey^="expanded"]'));
  log(`🔍 ${posts.length} post(s) found`);
  posts.forEach(processPost);
}

let scanTimeout = null;
function scheduleScan() {
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(findAndProcessPosts, 800);
}

const observer = new MutationObserver(scheduleScan);

async function init() {
  log("🚀 BSD v0.5 started");
  const stored = await chrome.storage.local.get([
    "bsd_mode", "bsd_threshold", "bsd_hide_sponsored", "bsd_silent_hide", "bsd_lang"
  ]);
  MODE           = stored.bsd_mode           || "filter";
  THRESHOLD      = stored.bsd_threshold      ?? 7;
  HIDE_SPONSORED = stored.bsd_hide_sponsored ?? true;
  SILENT_HIDE    = stored.bsd_silent_hide    ?? false;

  // Appliquer la langue persistée (override la détection Chrome)
  if (stored.bsd_lang === "fr" || stored.bsd_lang === "en") {
    LANG = stored.bsd_lang;
    log("🌐 Lang from storage:", LANG);
  }

  await loadModel();
  findAndProcessPosts();
  observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
