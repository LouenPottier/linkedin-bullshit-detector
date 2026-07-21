# CLAUDE.md

Guide pour agents IA travaillant sur ce dépôt.

## Vue d'ensemble

**LinkedIn Bullshit Detector (LBD)** est une extension Chrome (Manifest V3) qui note les posts LinkedIn de 0 à 10 ("bullshit score") et, selon le mode, les masque ou affiche un widget de notation pour collecter des labels. Le modèle est une **régression Ridge** (TF-IDF + features numériques) qui tourne **entièrement dans le navigateur** — aucune dépendance externe, aucun appel réseau, aucun build. Le ré-entraînement se fait en un clic depuis le popup, en JavaScript pur.

Tout est stocké localement via `chrome.storage.local`.

## Lancer / tester

Pas de build, pas de package manager. Pour tester :
1. `chrome://extensions` → activer **Developer mode** → **Load unpacked** → sélectionner ce dossier.
2. Après modification d'un fichier, recharger l'extension (bouton ↻) puis recharger linkedin.com.
3. Debug : le content script logge sous le préfixe `[BSD]` (constante `DEBUG` en haut de `content.js`).

## Fichiers

| Fichier | Rôle |
|---------|------|
| `manifest.json` | Manifest V3. Content script = `i18n.js` + `content.js`, injectés sur `https://www.linkedin.com/*`. `tfidf_vocab.json` exposé en `web_accessible_resources`. |
| `content.js` | **Cœur de l'extension côté page.** Scanne le feed, extrait les données des posts du DOM, calcule le score, applique le mode filtre/collecte. **Inference uniquement** (ne ré-entraîne pas). |
| `popup.js` | UI du popup + **pipeline de ré-entraînement complet en JS** (TF-IDF/IDF, StandardScaler, Ridge analytique). Lit/écrit les réglages et le modèle custom. |
| `popup.html` / `styles.css` | UI du popup et styles des widgets injectés (`.bsd-*`). |
| `i18n.js` | Traductions FR/EN. Définit la globale `LANG` et la fonction `t(key, ...args)`. Chargé dans les deux contextes (popup et content). |
| `tfidf_vocab.json` | **Modèle de base** (bootstrap) : vocabulaire, IDF, coefficients Ridge, params du scaler, top-emojis, pente sigmoïde. Généré par `training/train.py`. |
| `training/train.py` | Script Python (scikit-learn) qui régénère `tfidf_vocab.json` à partir d'un dataset JSON. Pour reproductibilité ; pas nécessaire en usage normal. |

## Modèle de scoring

Score = `sigmoid10(slope · (Ridge·features))`, borné [0, 10]. Features :
- **TF-IDF** : sac de mots/bigrammes pondéré sur `text + headline`, 500 features (`n_tfidf_features`).
- **Numériques** : `likes, comments, text_len, word_count, emoji_ratio, headline_len` + 4 features de phrases courtes (`short_sent_3/5/7/10`) + occurrences des top-10 emojis du corpus. Standardisées via StandardScaler.

Les labels sont transformés en `logit10` avant l'entraînement, et la prédiction repasse par `sigmoid10`.

### Deux modèles, un blend
- `BASE_MODEL` : toujours chargé depuis `tfidf_vocab.json`. Le modèle de base a **moins de features numériques** (pas de top-emojis appris) ; le code tronque à `n_num_features` et passe `topEmojis = null`.
- `CUSTOM_MODEL` : entraîné par l'utilisateur, stocké dans `chrome.storage.local` (`bsd_custom_model`).
- Le score final est un **blend pondéré** : `w·custom + (1-w)·base`, où `w = computeW(maeCustom, maeBase, nLabelled)` dépend du nombre de labels et de l'amélioration de MAE. **Cette fonction `computeW` est dupliquée à l'identique dans `content.js` et `popup.js` — garder les deux synchronisées.**

## Invariants critiques (à respecter)

La logique de features et de TF-IDF est **réimplémentée 3 fois** (`content.js`, `popup.js`, `train.py`). Toute modification d'une feature, du tokenizer, de l'ordre des features, ou des constantes (alpha, slope, top-N emojis, seuils de phrases courtes) **doit être répercutée dans les trois**, sinon entraînement et inference divergent silencieusement.

Fonctions à garder cohérentes entre les fichiers :
- `tokenize`, `tfidfVector(FromModel)` — même regex, même calcul TF-IDF (sublinear_tf, normalisation L2).
- `buildNumFeatures` / `build_features` — même ordre et même définition des features numériques.
- `hashPostId` / `hash_post_id` — **djb2 signé int32**, utilisé pour le split train/val déterministe (`hash % 5 == 0` → validation). Doit être identique en JS et Python.
- `logit10` / `sigmoid10`, `computeW`, `adaptiveAlpha`.

## Communication popup ↔ content script

Le popup envoie des messages via `chrome.tabs.sendMessage` ; `content.js` écoute dans `chrome.runtime.onMessage`. Types : `BSD_MODE_CHANGED`, `BSD_THRESHOLD_CHANGED`, `BSD_SPONSORED_CHANGED`, `BSD_SILENT_CHANGED`, `BSD_MODEL_UPDATED`, `BSD_MODEL_RESET`, `BSD_LANG_CHANGED`. La plupart déclenchent `resetAllPosts()` (retraitement du feed).

## Clés `chrome.storage.local`

`bullshit_dataset` (posts labellisés, indexés par postId), `bsd_custom_model`, `bsd_mae_base`, `bsd_mode`, `bsd_threshold`, `bsd_hide_sponsored`, `bsd_silent_hide`, `bsd_lang`, `bsd_stats` (compteurs de masquage).

## Extraction DOM (fragile)

`extractPostData` dans `content.js` dépend de la structure DOM de LinkedIn (attributs `componentkey`, libellés `aria-label` FR/EN, filtrage par listes `NOISE`). C'est le point le plus susceptible de casser quand LinkedIn change son interface. Les posts sont repérés via `[componentkey^="expanded"]`.

### Détection des posts sponsorisés

`isSponsored` scanne toutes les **feuilles textuelles courtes** (≤ 30 caractères, n'importe quelle balise, plus un repli sur `aria-label`) et les teste contre `SPONSORED_RE`, après normalisation (minuscules, accents retirés, puces de fin supprimées).

**Ne jamais revenir à une égalité stricte ni à un sélecteur d'attribut** : LinkedIn a déjà cassé la détection deux fois de cette manière. Historique :

| Avant | Après (juil. 2026) |
|-------|--------------------|
| `p[componentkey]` / `span[componentkey]` dont le `textContent` vaut exactement `"Sponsorisé"` | `<p componentkey><span>Post sponsorisé</span></p>` — libellé renommé **et** déplacé dans un `<span>` enfant |

Le motif est ancré aux deux bouts (`^…$`) pour qu'un vrai post parlant de publicité ne soit pas masqué à tort.

Pour diagnostiquer une future régression : helper `__bsdDumpLabels()` en fin de `content.js`, à lancer dans la console DevTools de linkedin.com **après avoir basculé le sélecteur de contexte JS sur l'extension** (le content script vit dans un monde isolé). Il liste, pour chaque post, les textes courts que voit le détecteur.

## Conventions

- Commentaires et messages utilisateur en **français** ; identifiants en anglais.
- Toute chaîne visible passe par `t(...)` (i18n FR/EN) — ne pas hardcoder de texte dans le DOM.
- Pas de framework, pas de dépendances JS. JS vanilla.
