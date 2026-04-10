// popup.js — v0.3

const STORAGE_KEYS = ["bullshit_dataset", "bsd_mode", "bsd_threshold"];

chrome.storage.local.get(STORAGE_KEYS, (result) => {
  const dataset   = result.bullshit_dataset || {};
  const mode      = result.bsd_mode || "filter";       // "filter" | "collect"
  const threshold = result.bsd_threshold ?? 7;

  // ── Mode toggle ──
  const btnFilter  = document.getElementById("btn-mode-filter");
  const btnCollect = document.getElementById("btn-mode-collect");
  const thresholdSection = document.getElementById("threshold-section");

  function applyMode(m) {
    btnFilter.classList.toggle("active", m === "filter");
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
  const slider   = document.getElementById("threshold-slider");
  const threshVal = document.getElementById("threshold-val");

  slider.value = threshold;
  threshVal.textContent = threshold + "/10";

  slider.addEventListener("input", () => {
    const v = parseInt(slider.value);
    threshVal.textContent = v + "/10";
    chrome.storage.local.set({ bsd_threshold: v });
    notifyContentScript({ type: "BSD_THRESHOLD_CHANGED", threshold: v });
  });

  function getCurrentThreshold() {
    return parseInt(slider.value);
  }

  // ── Stats collecte ──
  const posts = Object.values(dataset);

  if (posts.length === 0) {
    document.getElementById("empty-state").style.display = "block";
    document.getElementById("loaded-state").style.display = "none";
  } else {
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("loaded-state").style.display = "block";

    const scores  = posts.map(p => p.manualScore).filter(s => s !== undefined && s !== null);
    const avg     = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : "—";
    const high    = scores.filter(s => s >= 7).length;

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

    document.getElementById("btn-clear").addEventListener("click", () => {
      if (confirm(`Effacer les ${posts.length} posts sauvegardés ? Cette action est irréversible.`)) {
        chrome.storage.local.remove(["bullshit_dataset"], () => window.close());
      }
    });
  }
});

// Envoie un message au content script de l'onglet LinkedIn actif
function notifyContentScript(msg) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, msg).catch(() => {});
    }
  });
}
