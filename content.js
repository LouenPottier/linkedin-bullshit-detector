// ============================================================
// LinkedIn Bullshit Detector — content.js (v0.4 - Ridge model)
// ============================================================

const DEBUG = true;
const log = (...args) => DEBUG && console.log("[BSD]", ...args);

// ============================================================
// MOTEUR TF-IDF + RIDGE — inférence in-browser
// Le vocabulaire et les poids sont chargés depuis tfidf_vocab.json
// ============================================================

let MODEL = null; // sera peuplé par loadModel()

async function loadModel() {
  const url = chrome.runtime.getURL("tfidf_vocab.json");
  const res = await fetch(url);
  MODEL = await res.json();
  log("🧠 Modèle chargé —", MODEL.n_tfidf_features, "features TF-IDF +", MODEL.n_num_features, "numériques");
}

// Tokenisation identique à scikit-learn TfidfVectorizer par défaut :
// découpe sur les caractères non-alphanumériques, lowercase, tokens >= 2 chars
function tokenize(text) {
  return (text || "").toLowerCase().match(/[a-z0-9\u00c0-\u017e]+/g) || [];
}

// Construit le vecteur TF-IDF (unigrammes + bigrammes) pour un texte donné
function tfidfVector(text) {
  const tokens = tokenize(text);
  const vocab = MODEL.vocabulary;
  const idf = MODEL.idf;
  const n = MODEL.n_tfidf_features;
  const tf = new Float64Array(n);

  // Unigrammes
  for (const tok of tokens) {
    if (tok in vocab) tf[vocab[tok]] += 1;
  }
  // Bigrammes
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = tokens[i] + " " + tokens[i + 1];
    if (bigram in vocab) tf[vocab[bigram]] += 1;
  }

  // sublinear_tf : remplace tf par 1 + log(tf)
  for (let i = 0; i < n; i++) {
    if (tf[i] > 0) tf[i] = 1 + Math.log(tf[i]);
  }

  // Multiply by IDF
  for (let i = 0; i < n; i++) {
    tf[i] *= idf[i];
  }

  // Normalisation L2
  let norm = 0;
  for (let i = 0; i < n; i++) norm += tf[i] * tf[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < n; i++) tf[i] /= norm;

  return tf;
}

// Normalise les features numériques avec le scaler appris
function scaleNumFeatures(raw) {
  const mean  = MODEL.scaler_mean;
  const scale = MODEL.scaler_scale;
  return raw.map((v, i) => (v - mean[i]) / scale[i]);
}

function parseCount(s) {
  if (!s) return 0;
  const digits = String(s).replace(/[^\d]/g, "");
  return digits ? parseInt(digits, 10) : 0;
}

function feedContextFlag(fc) {
  return fc && fc.toLowerCase().includes("profil") ? 1 : 0;
}

// Score Ridge = dot(features, coef) + intercept, clampé [0, 10]
function computeModelScore(postData) {
  if (!MODEL) return { score: 0, found: [] };

  const combined = `${postData.text || ""} ${postData.headline || ""}`.trim();
  const tfidfVec = tfidfVector(combined);

  // Features numériques : ["likes", "comments", "autoScore", "text_len", "word_count", "fc_flag", "kw_count"]
  const lower = (postData.text || "").toLowerCase();
  const found = [...new Set(BSD_RULES.keywords.filter(kw => lower.includes(kw.toLowerCase())))];
  const rawNum = [
    parseCount(postData.likes),
    parseCount(postData.comments),
    0,                               // autoScore = 0 (on ne l'a pas encore)
    (postData.text || "").length,
    (postData.text || "").split(/\s+/).length,
    feedContextFlag(postData.feedContext),
    found.length,
  ];
  const scaledNum = scaleNumFeatures(rawNum);

  // Produit scalaire : TF-IDF part
  const coef = MODEL.ridge_coef;
  let score = MODEL.ridge_intercept;
  for (let i = 0; i < MODEL.n_tfidf_features; i++) {
    score += tfidfVec[i] * coef[i];
  }
  // Numériques part
  for (let i = 0; i < MODEL.n_num_features; i++) {
    score += scaledNum[i] * coef[MODEL.n_tfidf_features + i];
  }

  // Clamp et arrondi
  score = Math.round(Math.max(0, Math.min(10, score)));
  return { score, found };
}

// ============================================================
// EXTRACTION DES DONNÉES
// ============================================================
function extractPostData(postEl) {
  const leaves = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0);
  const texts = leaves.map(el => el.innerText.trim());

  const NOISE = ["post du fil d'actualité", "en fonction de votre profil et de votre activité",
    "j'aime", "commenter", "republier", "envoyer", "suivre", "signaler", "…", "plus"];

  // Auteur
  let author = "";
  const menuBtn = postEl.querySelector('button[aria-label^="Ouvrir le menu de commandes pour le post de"]');
  if (menuBtn) {
    author = menuBtn.getAttribute("aria-label")
      .replace("Ouvrir le menu de commandes pour le post de", "").trim();
  }

  // Headline
  let headline = "";
  const authorIdx = texts.indexOf(author);
  if (authorIdx !== -1) {
    for (let i = authorIdx + 1; i < texts.length; i++) {
      const t = texts[i];
      if (/^•\s*\d/.test(t)) continue;
      if (NOISE.includes(t.toLowerCase())) continue;
      headline = t;
      break;
    }
  }

  // Texte du post
  let text = "";
  let maxLen = 80;
  postEl.querySelectorAll('p').forEach(el => {
    if (el.children.length > 5) return;
    const t = el.innerText.trim();
    if (NOISE.includes(t.toLowerCase())) return;
    if (t.length > maxLen) { maxLen = t.length; text = t; }
  });

  // Likes / commentaires
  const likesRaw = texts.find(t => t.toLowerCase().includes("réaction") || t.toLowerCase().includes("reaction"));
  const likes = likesRaw ? likesRaw.split("\n")[0].trim() : "0";

  const commentsRaw = texts.find(t => t.toLowerCase().includes("commentaire") || t.toLowerCase().includes("comment"));
  const comments = commentsRaw ? commentsRaw.split("\n")[0].trim() : "0";

  // URL
  const linkEl = postEl.querySelector('a[href*="/posts/"], a[href*="activity"], a[href*="/feed/update/"]');
  const postUrl = linkEl ? linkEl.href.split("?")[0] : "";

  // Feed context
  const allLeaves = [...postEl.querySelectorAll('p, span')]
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
async function savePost(postId, data, manualScore, autoScore, autoKeywords) {
  return new Promise((resolve) => {
    chrome.storage.local.get(["bullshit_dataset"], (result) => {
      const dataset = result.bullshit_dataset || {};
      dataset[postId] = {
        ...data, postId, manualScore, autoScore, autoKeywords,
        savedAt: new Date().toISOString(),
        pageUrl: window.location.href,
      };
      chrome.storage.local.set({ bullshit_dataset: dataset }, resolve);
    });
  });
}

// ============================================================
// WIDGET
// ============================================================
function createWidget(postEl, postId, postData) {
  const { score: autoScore, found: autoKeywords } = computeModelScore(postData);
  const isSuspect = autoScore >= 4;
  const autoLevel = autoScore >= 7 ? "🔴" : autoScore >= 4 ? "🟠" : "🟢";

  const kwHtml = autoKeywords.length > 0
    ? autoKeywords.slice(0, 6).map(k => `<em>${k}</em>`).join(" ") +
      (autoKeywords.length > 6 ? ` <span style="color:#aaa">+${autoKeywords.length - 6}</span>` : "")
    : `<span class="bsd-clean">✓ Aucun mot-clé bullshit</span>`;

  const isPromoted = postData.feedContext?.toLowerCase().includes("profil") ||
                     postData.feedContext?.toLowerCase().includes("activité");
  const promotedHtml = isPromoted
    ? `<div class="bsd-promoted">📢 Post promu : <em>${postData.feedContext}</em></div>`
    : "";

  const widget = document.createElement("div");
  widget.className = "bsd-widget" + (isSuspect ? " bsd-suspect" : "");
  widget.dataset.postId = postId;
  widget.innerHTML = `
    ${promotedHtml}
    <div class="bsd-auto-badge">
      <span>${autoLevel} Modèle : <strong>${autoScore}/10</strong></span>
      <div class="bsd-keywords">${kwHtml}</div>
    </div>
    <div class="bsd-slider-wrap">
      <label>Score manuel : <strong class="bsd-manual-val">—</strong></label>
      <div class="bsd-slider-row">
        <span class="bsd-tick">0</span>
        <input type="range" class="bsd-slider" min="0" max="10" step="1" value="5">
        <span class="bsd-tick">10</span>
        <button class="bsd-save-btn">💾 Sauvegarder</button>
      </div>
    </div>
    <div class="bsd-saved-msg" style="display:none">✅ Sauvegardé !</div>
  `;

  const slider = widget.querySelector(".bsd-slider");
  const manualVal = widget.querySelector(".bsd-manual-val");
  const savedMsg = widget.querySelector(".bsd-saved-msg");
  const saveBtn = widget.querySelector(".bsd-save-btn");

  slider.addEventListener("input", () => {
    manualVal.textContent = slider.value + "/10";
  });

  saveBtn.addEventListener("click", async () => {
    await savePost(postId, postData, parseInt(slider.value), autoScore, autoKeywords);
    savedMsg.style.display = "block";
    saveBtn.textContent = "✅ Sauvé";
    setTimeout(() => { savedMsg.style.display = "none"; saveBtn.textContent = "💾 Sauvegarder"; }, 2000);
  });

  return widget;
}

// ============================================================
// INJECTION
// ============================================================
function processPost(postEl) {
  if (postEl.dataset.bsdDone) return;
  postEl.dataset.bsdDone = "1";
  const data = extractPostData(postEl);
  const postId = generatePostId(postEl);
  log("✅", data.author || "(sans auteur)", "—", data.text.slice(0, 60));

  const widget = createWidget(postEl, postId, data);
  postEl.appendChild(widget);
}

function findAndProcessPosts() {
  const all = document.querySelectorAll('[componentkey^="expanded"]');
  const posts = [...all].filter(el => !el.parentElement?.closest('[componentkey^="expanded"]'));
  log(`🔍 ${posts.length} post(s) trouvé(s)`);
  posts.forEach(processPost);
}

// ============================================================
// OBSERVER + INIT
// ============================================================
let scanTimeout = null;
function scheduleScan() {
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(findAndProcessPosts, 800);
}

const observer = new MutationObserver(scheduleScan);

async function init() {
  log("🚀 BSD v0.4 démarré");
  await loadModel();
  findAndProcessPosts();
  observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
