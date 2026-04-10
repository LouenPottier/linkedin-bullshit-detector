// ============================================================
// LinkedIn Bullshit Detector — content.js (v0.3 - correct selectors)
// ============================================================

const DEBUG = true;
const log = (...args) => DEBUG && console.log("[BSD]", ...args);

// ============================================================
// MOTS-CLÉS
// ============================================================
// BSD_RULES est défini dans rules.js, chargé avant ce fichier.

function scoreHeadline(headline) {
  if (!headline) return { penalty: 0, reason: null };
  const lower = headline.toLowerCase();
  const reasons = [];
  let penalty = 0;

  // Pronoms personnels — toute headline avec un pronom est suspecte
  const pronouns = ["je ", "j'", "on ", "tu ", "vous ", "ton ", "votre ", "vos ",
                    "ils ", "elles ", "nous ", "i ", "we ", "you ", "your "];
  const hasPronoun = pronouns.some(p => lower.includes(p));
  if (hasPronoun) {
    penalty += BSD_RULES.headline.pronounPenalty;
    reasons.push("pronom personnel");
  }

  // Mots-clés bullshit spécifiques (™, néologismes, etc.)
  const hasWord = BSD_RULES.headline.bullshitWords.some(w => lower.includes(w.toLowerCase()));
  if (hasWord) {
    penalty += BSD_RULES.headline.keywordPenalty;
    reasons.push("mot-clé suspect");
  }

  // Headline longue (> 80 chars) = probablement un pitch commercial, pas un titre
  if (headline.length > 80) {
    penalty += BSD_RULES.headline.lengthPenalty;
    reasons.push("intitulé trop long");
  }

  // Présence de ":" suivi de liste de services
  if (headline.includes(":") && headline.length > 40) {
    penalty += BSD_RULES.headline.colonPenalty;
    reasons.push("liste de services");
  }

  return { penalty: Math.min(penalty, 6), reason: reasons.join(", ") || null };
}

function computeAutoScore(text, headline) {
  const lower = (text || "").toLowerCase();
  const found = [...new Set(BSD_RULES.keywords.filter(kw => lower.includes(kw.toLowerCase())))];
  const keywordScore = Math.round(Math.log1p(found.length) * 3.5);

  const { penalty, reason: headlineReason } = scoreHeadline(headline);

  const score = Math.min(10, keywordScore + penalty);
  return { score, found, headlineReason };
}

// ============================================================
// EXTRACTION DES DONNÉES
// ============================================================
function extractPostData(postEl) {
  // Nœuds feuilles : <p> et <span> sans enfants, plus les <p> avec peu d'enfants inline
  const leaves = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0);
  const texts = leaves.map(el => el.innerText.trim());

  const NOISE = ["post du fil d'actualité", "en fonction de votre profil et de votre activité",
    "j'aime", "commenter", "republier", "envoyer", "suivre", "signaler", "…", "plus"];

  // Auteur : via aria-label du bouton menu (fiable)
  let author = "";
  const menuBtn = postEl.querySelector('button[aria-label^="Ouvrir le menu de commandes pour le post de"]');
  if (menuBtn) {
    author = menuBtn.getAttribute("aria-label")
      .replace("Ouvrir le menu de commandes pour le post de", "").trim();
  }

  // Headline : texte après l'auteur, en ignorant les degrés de relation (• 2e, • 3e...)
  let headline = "";
  const authorIdx = texts.indexOf(author);
  if (authorIdx !== -1) {
    for (let i = authorIdx + 1; i < texts.length; i++) {
      const t = texts[i];
      if (/^•\s*\d/.test(t)) continue;           // "• 2e et +" → skip
      if (NOISE.includes(t.toLowerCase())) continue;
      headline = t;
      break;
    }
  }

  // Texte du post : le <p> avec peu d'enfants le plus long (> 80 chars)
  let text = "";
  let maxLen = 80;
  postEl.querySelectorAll('p').forEach(el => {
    if (el.children.length > 5) return;
    const t = el.innerText.trim();
    if (NOISE.includes(t.toLowerCase())) return;
    if (t.length > maxLen) { maxLen = t.length; text = t; }
  });

  // Likes : premier texte contenant "réaction"
  const likesRaw = texts.find(t => t.toLowerCase().includes("réaction") || t.toLowerCase().includes("reaction"));
  const likes = likesRaw ? likesRaw.split("\n")[0].trim() : "0";

  // Commentaires : premier texte contenant "commentaire"
  const commentsRaw = texts.find(t => t.toLowerCase().includes("commentaire") || t.toLowerCase().includes("comment"));
  const comments = commentsRaw ? commentsRaw.split("\n")[0].trim() : "0";

  // URL du post
  const linkEl = postEl.querySelector('a[href*="/posts/"], a[href*="activity"], a[href*="/feed/update/"]');
  const postUrl = linkEl ? linkEl.href.split("?")[0] : "";

  // Contexte du feed : toujours à l'index 1 dans les feuilles
  // index 0 = "Post du fil d'actualité", index 1 = origine du post
  const allLeaves = [...postEl.querySelectorAll('p, span')]
    .filter(el => el.children.length === 0 && el.innerText.trim().length > 0)
    .map(el => el.innerText.trim());
  const feedContext = (allLeaves[1] && allLeaves[1] !== author) ? allLeaves[1] : "";

  return { text, author, headline, likes, comments, postUrl, feedContext };
}

function generatePostId(postEl) {
  const key = postEl.getAttribute("componentkey") || "";
  if (key) return "post_" + key.slice(0, 20).replace(/[^a-z0-9]/gi, "");
  // Fallback hash
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
  const { score: autoScore, found: autoKeywords, headlineReason } = computeAutoScore(postData.text, postData.headline);
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
  const headlineHtml = headlineReason
    ? `<div class="bsd-headline-flag">⚠️ Intitulé suspect : <em>${(postData.headline || "").slice(0, 60)}</em></div>`
    : "";

  const widget = document.createElement("div");
  widget.className = "bsd-widget" + (isSuspect ? " bsd-suspect" : "");
  widget.dataset.postId = postId;
  widget.innerHTML = `
    ${promotedHtml}
    <div class="bsd-auto-badge">
      <span>${autoLevel} Auto : <strong>${autoScore}/10</strong></span>
      <div class="bsd-keywords">${kwHtml}</div>
    </div>
    ${headlineHtml}
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

  // Injecter à la fin du post (avant le dernier enfant ou à la fin)
  postEl.appendChild(widget);
}

function findAndProcessPosts() {
  const all = document.querySelectorAll('[componentkey^="expanded"]');
  // Garder uniquement les éléments qui ne sont PAS imbriqués dans un autre expanded
  const posts = [...all].filter(el => !el.parentElement?.closest('[componentkey^="expanded"]'));
  log(`🔍 ${posts.length} post(s) trouvé(s) (${all.length} au total, doublons filtrés)`);
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

function init() {
  log("🚀 BSD v0.3 démarré");
  findAndProcessPosts();
  observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
