// popup.js

chrome.storage.local.get(["bullshit_dataset"], (result) => {
  const dataset = result.bullshit_dataset || {};
  const posts = Object.values(dataset);

  if (posts.length === 0) {
    document.getElementById("empty-state").style.display = "block";
    document.getElementById("loaded-state").style.display = "none";
    return;
  }

  document.getElementById("empty-state").style.display = "none";
  document.getElementById("loaded-state").style.display = "block";

  // Stats
  const total = posts.length;
  const scores = posts.map(p => p.manualScore).filter(s => s !== undefined && s !== null);
  const avg = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : "—";
  const high = scores.filter(s => s >= 7).length;
  const authors = new Set(posts.map(p => p.author).filter(Boolean)).size;

  document.getElementById("stat-total").textContent = total;
  document.getElementById("stat-avg").textContent = avg;
  document.getElementById("stat-high").textContent = high;
  document.getElementById("stat-authors").textContent = authors;

  // Dernière sauvegarde
  const dates = posts.map(p => p.savedAt).filter(Boolean).sort();
  if (dates.length > 0) {
    const last = new Date(dates[dates.length - 1]);
    document.getElementById("last-saved").textContent =
      "Dernière sauvegarde : " + last.toLocaleDateString("fr-FR") + " " + last.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  }

  // Export JSON
  document.getElementById("btn-export").addEventListener("click", () => {
    const json = JSON.stringify(posts, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bullshit_dataset_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Clear
  document.getElementById("btn-clear").addEventListener("click", () => {
    if (confirm(`Effacer les ${total} posts sauvegardés ? Cette action est irréversible.`)) {
      chrome.storage.local.remove(["bullshit_dataset"], () => {
        window.close();
      });
    }
  });
});
