// ============================================================
// LinkedIn Bullshit Detector — content.js (v0.4)
// ============================================================

const DEBUG = true;
const log = (...args) => DEBUG && console.log("[BSD]", ...args);

// ── État global ──
let MODE           = "filter";   // "filter" | "collect"
let THRESHOLD      = 7;
let HIDE_SPONSORED = true;
let SILENT_HIDE    = false;
let MODEL          = null;

// ============================================================
// MODÈLE — TF-IDF + Ridge in-browser
// ============================================================

async function loadModel() {
  const url = chrome.runtime.getURL("tfidf_vocab.json");
  const res = await fetch(url);
  MODEL = await res.json();
  log("🧠 Modèle chargé —", MODEL.n_tfidf_features, "TF-IDF +", MODEL.n_num_features, "numériques");
}

function tokenize(text) {
  return (text || "").toLowerCase().match(/[a-z0-9\u00c0-\u017e]+/g) || [];
}

function tfidfVector(text) {
  const tokens = tokenize(text);
  const vocab  = MODEL.vocabulary;
  const idf    = MODEL.idf;
  const n      = MODEL.n_tfidf_features;
  const tf     = new Float64Array(n);

  for (const tok of tokens) {
    if (tok in vocab) tf[vocab[tok]] += 1;
  }
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = tokens[i] + " " + tokens[i + 1];
    if (bigram in vocab) tf[vocab[bigram]] += 1;
  }
  for (let i = 0; i < n; i++) {
    if (tf[i] > 0) tf[i] = 1 + Math.log(tf[i]);
  }
  for (let i = 0; i < n; i++) tf[i] *= idf[i];

  let norm = 0;
  for (let i = 0; i < n; i++) norm += tf[i] * tf[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < n; i++) tf[i] /= norm;

  return tf;
}

function scaleNumFeatures(raw) {
  return raw.map((v, i) => (v - MODEL.scaler_mean[i]) / MODEL.scaler_scale[i]);
}

function parseCount(s) {
  const digits = String(s || "").replace(/[^\d]/g, "");
  return digits ? parseInt(digits, 10) : 0;
}

function feedContextFlag(fc) {
  return fc && fc.toLowerCase().includes("profil") ? 1 : 0;
}

function computeModelScore(postData) {
  if (!MODEL) return { score: 0, found: [] };

  const combined = `${postData.text || ""} ${postData.headline || ""}`.trim();
  const tfidfVec = tfidfVector(combined);

  const lower = (postData.text || "").toLowerCase();
  const found = [...new Set(BSD_RULES.keywords.filter(kw => lower.includes(kw.toLowerCase())))];
  const textWords = (postData.text || "").split(/\s+/);
  const wordCount = textWords.length;
  const emojiCount = [...(postData.text || "")].filter(c => c.codePointAt(0) > 0x1F000).length;

  const rawNum = [
    parseCount(postData.likes),
    parseCount(postData.comments),
    0,
    (postData.text || "").length,
    wordCount,
    feedContextFlag(postData.feedContext),
    found.length,
    emojiCount / Math.max(wordCount, 1),
    (postData.headline || "").length,
  ];
  const scaledNum = scaleNumFeatures(rawNum);

  const coef = MODEL.ridge_coef;
  let score  = MODEL.ridge_intercept;
  for (let i = 0; i < MODEL.n_tfidf_features; i++) score += tfidfVec[i] * coef[i];
  for (let i = 0; i < MODEL.n_num_features; i++)   score += scaledNum[i] * coef[MODEL.n_tfidf_features + i];

  score = Math.round(Math.max(0, Math.min(10, score)));
  return { score, found };
}

// ============================================================
// DÉTECTION — posts sponsorisés
// ============================================================

function isSponsored(postEl) {
  return [...postEl.querySelectorAll('p[componentkey], span[componentkey]')]
    .some(el => el.textContent.trim() === 'Sponsorisé');
}

// ============================================================
// EXTRACTION
// ============================================================

function extractPostData(postEl) {
  const leaves = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0);
  const texts = leaves.map(el => el.innerText.trim());

  const NOISE = ["post du fil d'actualité", "en fonction de votre profil et de votre activité",
    "j'aime", "commenter", "republier", "envoyer", "suivre", "signaler", "…", "plus"];

  let author = "";
  const menuBtn = postEl.querySelector('button[aria-label^="Ouvrir le menu de commandes pour le post de"]');
  if (menuBtn) {
    author = menuBtn.getAttribute("aria-label")
      .replace("Ouvrir le menu de commandes pour le post de", "").trim();
  }

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

  let text = "", maxLen = 80;
  postEl.querySelectorAll('p').forEach(el => {
    if (el.children.length > 5) return;
    const t = el.innerText.trim();
    if (NOISE.includes(t.toLowerCase())) return;
    if (t.length > maxLen) { maxLen = t.length; text = t; }
  });

  const likesRaw    = texts.find(t => t.toLowerCase().includes("réaction") || t.toLowerCase().includes("reaction"));
  const likes       = likesRaw ? likesRaw.split("\n")[0].trim() : "0";
  const commentsRaw = texts.find(t => t.toLowerCase().includes("commentaire") || t.toLowerCase().includes("comment"));
  const comments    = commentsRaw ? commentsRaw.split("\n")[0].trim() : "0";

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

async function savePost(postId, data, manualScore, autoScore, autoKeywords) {
  return new Promise((resolve) => {
    chrome.storage.local.get(["bullshit_dataset"], (result) => {
      const dataset = result.bullshit_dataset || {};
      const { author, ...dataWithoutAuthor } = data;
      dataset[postId] = {
        ...dataWithoutAuthor, postId, manualScore, autoScore, autoKeywords,
        savedAt: new Date().toISOString(),
        pageUrl: window.location.href,
      };
      chrome.storage.local.set({ bullshit_dataset: dataset }, resolve);
    });
  });
}

// ============================================================
// MODE FILTRE — placeholder cliquable
// ============================================================

function applyFilterMode(postEl, score, sponsored = false) {
  postEl.querySelectorAll(':scope > *').forEach(el => {
    if (!el.classList.contains('bsd-placeholder')) el.style.display = "none";
  });

  if (postEl.querySelector('.bsd-placeholder')) return;

  if (SILENT_HIDE) return;

  const label = sponsored
    ? "📢 Post sponsorisé masqué"
    : `${score >= 8 ? "🔴" : "🟠"} Post masqué — score bullshit : <strong>${score}/10</strong>`;

  const placeholder = document.createElement("div");
  placeholder.className = "bsd-placeholder" + (sponsored ? " bsd-placeholder-sponsored" : "");
  placeholder.innerHTML = `
    <span>${label}</span>
    <button class="bsd-show-btn">Afficher quand même</button>
  `;

  placeholder.querySelector(".bsd-show-btn").addEventListener("click", () => {
    placeholder.remove();
    postEl.querySelectorAll(':scope > *').forEach(el => el.style.display = "");
    postEl.dataset.bsdDone = "shown";
  });

  postEl.appendChild(placeholder);
}

// ============================================================
// MODE COLLECTE — widget notation
// ============================================================

function createCollectWidget(postEl, postId, postData, autoScore, autoKeywords) {
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

  const slider    = widget.querySelector(".bsd-slider");
  const manualVal = widget.querySelector(".bsd-manual-val");
  const savedMsg  = widget.querySelector(".bsd-saved-msg");
  const saveBtn   = widget.querySelector(".bsd-save-btn");

  slider.addEventListener("input", () => { manualVal.textContent = slider.value + "/10"; });

  saveBtn.addEventListener("click", async () => {
    await savePost(postId, postData, parseInt(slider.value), autoScore, autoKeywords);
    savedMsg.style.display = "block";
    saveBtn.textContent = "✅ Sauvé";
    setTimeout(() => { savedMsg.style.display = "none"; saveBtn.textContent = "💾 Sauvegarder"; }, 2000);
  });

  return widget;
}

// ============================================================
// TRAITEMENT D'UN POST
// ============================================================

function processPost(postEl) {
  if (postEl.dataset.bsdDone === "shown") return;
  if (postEl.dataset.bsdDone === "1") return;

  // Vérifier que c'est bien un post (et pas une recommandation de contacts, etc.)
  const menuBtn = postEl.querySelector('button[aria-label^="Ouvrir le menu de commandes pour le post de"]');
  if (!menuBtn) return;

  postEl.dataset.bsdDone = "1";

  // ── Posts sponsorisés ──
  if (MODE === "filter" && HIDE_SPONSORED && isSponsored(postEl)) {
    log("📢 Sponsorisé masqué");
    applyFilterMode(postEl, 0, true);
    return;
  }

  const data            = extractPostData(postEl);
  const postId          = generatePostId(postEl);
  const { score, found } = computeModelScore(data);

  log("✅", data.author || "(sans auteur)", `[${score}/10]`, "—", data.text.slice(0, 50));

  if (MODE === "filter" && score >= THRESHOLD) {
    applyFilterMode(postEl, score);
  } else if (MODE === "collect") {
    postEl.appendChild(createCollectWidget(postEl, postId, data, score, found));
  }
}

// ============================================================
// CHANGEMENT DE MODE — reset propre
// ============================================================

function resetAllPosts() {
  document.querySelectorAll('[data-bsd-done]').forEach(el => {
    el.querySelectorAll('.bsd-widget, .bsd-placeholder').forEach(w => w.remove());
    el.querySelectorAll(':scope > *').forEach(c => c.style.display = "");
    delete el.dataset.bsdDone;
  });
  findAndProcessPosts();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "BSD_MODE_CHANGED") {
    MODE = msg.mode;
    if (msg.threshold !== undefined) THRESHOLD = msg.threshold;
    log("🔄 Mode →", MODE, "/ seuil →", THRESHOLD);
    resetAllPosts();
  }
  if (msg.type === "BSD_THRESHOLD_CHANGED") {
    THRESHOLD = msg.threshold;
    log("🔄 Seuil →", THRESHOLD);
    resetAllPosts();
  }
  if (msg.type === "BSD_SPONSORED_CHANGED") {
    HIDE_SPONSORED = msg.hideSponsored;
    log("🔄 Sponsorisés →", HIDE_SPONSORED ? "masqués" : "visibles");
    resetAllPosts();
  }
  if (msg.type === "BSD_SILENT_CHANGED") {
    SILENT_HIDE = msg.silent;
    log("🔄 Silencieux →", SILENT_HIDE);
    resetAllPosts();
  }
});

// ============================================================
// SCAN + OBSERVER
// ============================================================

function findAndProcessPosts() {
  const all   = document.querySelectorAll('[componentkey^="expanded"]');
  const posts = [...all].filter(el => !el.parentElement?.closest('[componentkey^="expanded"]'));
  log(`🔍 ${posts.length} post(s) trouvé(s)`);
  posts.forEach(processPost);
}

let scanTimeout = null;
function scheduleScan() {
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(findAndProcessPosts, 800);
}

const observer = new MutationObserver(scheduleScan);

async function init() {
  log("🚀 BSD v0.5 démarré");
  const stored   = await chrome.storage.local.get(["bsd_mode", "bsd_threshold", "bsd_hide_sponsored", "bsd_silent_hide"]);
  MODE           = stored.bsd_mode          || "filter";
  THRESHOLD      = stored.bsd_threshold     ?? 7;
  HIDE_SPONSORED = stored.bsd_hide_sponsored ?? true;
  SILENT_HIDE    = stored.bsd_silent_hide   ?? false;
  await loadModel();
  findAndProcessPosts();
  observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
