// ============================================================
// LinkedIn Bullshit Detector — i18n.js
// Fonctionne dans deux contextes :
//   • Popup       : LANG lu depuis sessionStorage (synchrone)
//   • Content script : LANG initialisé via chrome.i18n,
//                      puis mis à jour par BSD_LANG_CHANGED
// ============================================================

function _detectLang() {
  // sessionStorage disponible uniquement dans le popup
  try {
    const saved = sessionStorage.getItem("bsd_lang");
    if (saved === "fr" || saved === "en") return saved;
  } catch (e) {}
  // Langue Chrome (fonctionne dans les deux contextes)
  try {
    const ui = chrome.i18n.getUILanguage();
    if (ui && ui.toLowerCase().startsWith("fr")) return "fr";
  } catch (e) {}
  return "en";
}

// `let` pour permettre la mise à jour par le content script
let LANG = _detectLang();

const STRINGS = {
  fr: {
    popup_subtitle:        "LinkedIn — détection & collecte v0.5",
    popup_mode_label:      "Mode",
    popup_mode_filter:     "🚫 Filtre",
    popup_mode_collect:    "🧪 Collecte",
    popup_threshold_label: "Seuil de masquage",
    popup_threshold_hint:  "Posts avec un score ≥ au seuil seront masqués.",
    popup_sponsored_label: "📢 Masquer les sponsorisés",
    popup_silent_label:    "🔇 Masquage silencieux",
    popup_empty:           "Aucun post sauvegardé pour l'instant.",
    popup_stat_total:      "Posts sauvegardés",
    popup_stat_avg:        "Score moyen",
    popup_stat_high:       "Score ≥ 7 (🔴)",
    popup_last_saved:      (date, time) => `Dernière sauvegarde : ${date} ${time}`,
    popup_export:          "⬇️ Exporter JSON",
    popup_clear:           "🗑️ Effacer les données",
    popup_clear_confirm:   (n) => `Effacer les ${n} posts sauvegardés ? Cette action est irréversible.`,
    train_title:           "🔁 Modèle personnalisé",
    train_badge_base:      "Base",
    train_badge_custom:    "Personnalisé ✓",
    train_desc_base:       "Réentraîne Ridge sur tes labels.",
    train_desc_custom:     (n, mae) => `${n} posts · MAE ${mae} pts`,
    train_btn:             "🔁 Réentraîner le modèle",
    train_btn_min:         (n, min) => `🔁 Réentraîner (${n}/${min} min.)`,
    train_btn_validating:  "⏳ Validation…",
    train_btn_training:    "⏳ Entraînement…",
    train_progress_val:    "Validation…",
    train_progress_final:  "Finalisation…",
    train_progress_done:   "Terminé",
    train_reset:           "↩️ Revenir au modèle de base",
    train_error_vocab:     "Erreur : impossible de charger tfidf_vocab.json",
    train_error_generic:   (msg) => `Erreur pendant l'entraînement : ${msg}`,
    train_warn_few:        (n) => `⚠️ Seulement ${n} posts labellisés.\n\nEntraîner un modèle sur aussi peu de données est très déconseillé.\n\nIl est recommandé d'atteindre au moins 50 posts avant d'entraîner.\n\nContinuer quand même ?`,
    content_sponsored:     "📢 Post sponsorisé masqué",
    content_hidden:        (score) => `${score >= 8 ? "🔴" : "🟠"} Post masqué — score bullshit : <strong>${score}/10</strong>`,
    content_show:          "Afficher quand même",
    widget_model_score:    (score) => `Modèle : <strong>${score}/10</strong>`,
    widget_no_keywords:    "✓ Aucun mot-clé bullshit",
    widget_manual_label:   "Score manuel :",
    widget_manual_none:    "—",
    widget_save:           "💾 Sauvegarder",
    widget_saved:          "✅ Sauvé",
    widget_saved_msg:      "✅ Sauvegardé !",
    widget_promoted:       (ctx) => `📢 Post promu : <em>${ctx}</em>`,
  },

  en: {
    popup_subtitle:        "LinkedIn — detection & collection v0.5",
    popup_mode_label:      "Mode",
    popup_mode_filter:     "🚫 Filter",
    popup_mode_collect:    "🧪 Collect",
    popup_threshold_label: "Hiding threshold",
    popup_threshold_hint:  "Posts with a score ≥ threshold will be hidden.",
    popup_sponsored_label: "📢 Hide sponsored posts",
    popup_silent_label:    "🔇 Silent hiding",
    popup_empty:           "No saved posts yet.",
    popup_stat_total:      "Saved posts",
    popup_stat_avg:        "Average score",
    popup_stat_high:       "Score ≥ 7 (🔴)",
    popup_last_saved:      (date, time) => `Last saved: ${date} ${time}`,
    popup_export:          "⬇️ Export JSON",
    popup_clear:           "🗑️ Clear data",
    popup_clear_confirm:   (n) => `Delete ${n} saved posts? This cannot be undone.`,
    train_title:           "🔁 Custom model",
    train_badge_base:      "Base",
    train_badge_custom:    "Custom ✓",
    train_desc_base:       "Retrain Ridge on your labels.",
    train_desc_custom:     (n, mae) => `${n} posts · MAE ${mae} pts`,
    train_btn:             "🔁 Retrain model",
    train_btn_min:         (n, min) => `🔁 Retrain (${n}/${min} min.)`,
    train_btn_validating:  "⏳ Validating…",
    train_btn_training:    "⏳ Training…",
    train_progress_val:    "Validating…",
    train_progress_final:  "Finalising…",
    train_progress_done:   "Done",
    train_reset:           "↩️ Revert to base model",
    train_error_vocab:     "Error: could not load tfidf_vocab.json",
    train_error_generic:   (msg) => `Training error: ${msg}`,
    train_warn_few:        (n) => `⚠️ Only ${n} labelled posts.\n\nTraining on so few samples is not recommended.\n\nAt least 50 posts are recommended before training.\n\nContinue anyway?`,
    content_sponsored:     "📢 Sponsored post hidden",
    content_hidden:        (score) => `${score >= 8 ? "🔴" : "🟠"} Post hidden — bullshit score: <strong>${score}/10</strong>`,
    content_show:          "Show anyway",
    widget_model_score:    (score) => `Model: <strong>${score}/10</strong>`,
    widget_no_keywords:    "✓ No bullshit keywords",
    widget_manual_label:   "Manual score:",
    widget_manual_none:    "—",
    widget_save:           "💾 Save",
    widget_saved:          "✅ Saved",
    widget_saved_msg:      "✅ Saved!",
    widget_promoted:       (ctx) => `📢 Promoted post: <em>${ctx}</em>`,
  },
};

function t(key, ...args) {
  const val = STRINGS[LANG][key];
  if (val === undefined) {
    console.warn("[BSD i18n] Missing key:", key);
    return key;
  }
  return typeof val === "function" ? val(...args) : val;
}
