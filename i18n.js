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
    popup_subtitle:        "Filtre de contenu LinkedIn",
    popup_mode_label:      "Mode",
    popup_mode_filter:     "🚫 Filtre",
    popup_mode_collect:    "🧪 Collecte",
    popup_threshold_label: "Seuil de tolérance au bullshit",
    popup_threshold_hint:  "Les posts avec un score ≥ au seuil seront masqués.",
    popup_sponsored_label: "📢 Masquer les posts sponsorisés",
    popup_silent_label:    "🔇 Masquage silencieux",
    popup_empty:           "Aucun post sauvegardé pour l'instant.",
    popup_stat_total:      "Posts sauvegardés",
    popup_stat_avg:        "Score moyen",
    popup_stat_high:       "Score ≥ 7 (🔴)",
    popup_stat_model:      "Personnalisation",
    popup_last_saved:      (date, time) => `Dernière sauvegarde : ${date} ${time}`,
    popup_import:          "⬆️ Importer JSON",
    popup_import_confirm:  (n, m) => `Importer ${n} posts ? Ils seront fusionnés avec tes ${m} posts existants (les doublons seront ignorés).`,
    popup_import_done:     (n) => `${n} nouveaux posts importés.`,
    popup_import_error:    "Fichier invalide ou corrompu.",
    popup_export:          "⬇️ Exporter JSON",
    popup_clear:           "🗑️ Supprimer la collecte",
    popup_clear_confirm:   (n) => `Supprimer les ${n} posts sauvegardés ? Cette action est irréversible.`,
    popup_share_note:      "Partage ta collecte avec tes amis pour améliorer leur bullshitomètre !",
    popup_dev_note:        "Outil de développement — les données restent locales.",
    train_title:           "🔁 Modèle personnalisé",
    train_badge_base:      "Base",
    train_badge_custom:    "Personnalisé ✓",
    train_desc_base:       "Réentraîne Ridge sur tes labels.",
    train_desc_custom:     (n) => `${n} posts labellisés`,
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
    content_silent_btn:    "🔇 Ne plus afficher ces encadrés",
    widget_auto_label:     "Bullshit détecté :",
    widget_model_score:    (score) => `<strong>${score}/10</strong>`,
    widget_manual_label:   "Votre note :",
    widget_manual_none:    "—",
    widget_scale_low:      "Pertinent",
    widget_scale_high:     "Bullshit",
    widget_save:           "💾 Sauvegarder",
    widget_saved:          "✅ Sauvé",
    widget_saved_msg:      "✅ Sauvegardé !",
    widget_promoted:       (ctx) => `📢 Post promu : <em>${ctx}</em>`,
    model_quality_very_low:  "Très faible",
    model_quality_low:       "Faible",
    model_quality_medium:    "Moyenne",
    model_quality_high:      "Forte",
    model_quality_very_high: "Très forte",
    model_quality_optimal:   "Optimale",
    model_quality_none:      "Nulle",
    model_quality_no_data:   "Note des posts pour l'évaluer !",
    filter_stat_today_bs:       "Posts bullshit masqués aujourd'hui",
    filter_stat_today_sponsored:"Posts sponsorisés masqués aujourd'hui",
    filter_stat_total:          "Posts masqués au total",
    filter_stat_scroll:         "Scroll économisé",
    popup_filter_update_note:   "Les posts ne sont plus masqués ? LinkedIn a peut-être modifié son site. Vérifie si une version plus récente de l'extension est disponible :",
    popup_filter_update_btn:    "Voir sur GitHub →",
  },

  en: {
    popup_subtitle:        "LinkedIn content filter",
    popup_mode_label:      "Mode",
    popup_mode_filter:     "🚫 Filter",
    popup_mode_collect:    "🧪 Collect",
    popup_threshold_label: "Bullshit tolerance threshold",
    popup_threshold_hint:  "Posts with a score ≥ threshold will be hidden.",
    popup_sponsored_label: "📢 Hide sponsored posts",
    popup_silent_label:    "🔇 Silent hiding",
    popup_empty:           "No saved posts yet.",
    popup_stat_total:      "Saved posts",
    popup_stat_avg:        "Average score",
    popup_stat_high:       "Score ≥ 7 (🔴)",
    popup_stat_model:      "Personalisation",
    popup_last_saved:      (date, time) => `Last saved: ${date} ${time}`,
    popup_import:          "⬆️ Import JSON",
    popup_import_confirm:  (n, m) => `Import ${n} posts? They will be merged with your ${m} existing posts (duplicates will be skipped).`,
    popup_import_done:     (n) => `${n} new posts imported.`,
    popup_import_error:    "Invalid or corrupted file.",
    popup_export:          "⬇️ Export JSON",
    popup_clear:           "🗑️ Delete collection",
    popup_clear_confirm:   (n) => `Delete ${n} saved posts? This cannot be undone.`,
    popup_share_note:      "Share your collection with friends to improve their bullshit detector!",
    popup_dev_note:        "Development tool — data stays local.",
    train_title:           "🔁 Custom model",
    train_badge_base:      "Base",
    train_badge_custom:    "Custom ✓",
    train_desc_base:       "Retrain Ridge on your labels.",
    train_desc_custom:     (n) => `${n} labelled posts`,
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
    content_silent_btn:    "🔇 Stop showing these banners",
    widget_auto_label:     "Detected bullshit:",
    widget_model_score:    (score) => `<strong>${score}/10</strong>`,
    widget_manual_label:   "Your score:",
    widget_manual_none:    "—",
    widget_scale_low:      "Relevant",
    widget_scale_high:     "Bullshit",
    widget_save:           "💾 Save",
    widget_saved:          "✅ Saved",
    widget_saved_msg:      "✅ Saved!",
    widget_promoted:       (ctx) => `📢 Promoted post: <em>${ctx}</em>`,
    model_quality_very_low:  "Very low",
    model_quality_low:       "Low",
    model_quality_medium:    "Medium",
    model_quality_high:      "High",
    model_quality_very_high: "Very high",
    model_quality_optimal:   "Optimal",
    model_quality_none:      "None",
    model_quality_no_data:   "Rate some posts to find out!",
    filter_stat_today_bs:       "Bullshit posts hidden today",
    filter_stat_today_sponsored:"Sponsored posts hidden today",
    filter_stat_total:          "Total posts hidden",
    filter_stat_scroll:         "Scroll saved",
    popup_filter_update_note:   "Posts no longer being hidden? LinkedIn may have updated its site. Check if a newer version of the extension is available:",
    popup_filter_update_btn:    "View on GitHub →",
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
