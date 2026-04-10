# Training — LinkedIn Bullshit Detector

Ce dossier contient le script d'entraînement du modèle de scoring automatique.

Le modèle est une **Ridge Regression** entraînée sur des features TF-IDF (texte du post + intitulé de l'auteur) et des features numériques (likes, commentaires, longueur du texte, ratio d'emojis…). Une fois entraîné, il est exporté sous forme de `tfidf_vocab.json` — un fichier JSON léger (~34 KB) qui contient le vocabulaire, les valeurs IDF, les coefficients Ridge et les paramètres de normalisation. L'inférence est ensuite faite directement dans le navigateur sans aucune dépendance externe.

Paramètres actuels : `alpha=0.1`, `min_df=4`, `max_features=500`, bigrammes activés.

---

## Prérequis

Python 3.9+ et pip :

```bash
pip install scikit-learn onnx skl2onnx
```

---

## Collecter des données

Utilise l'extension **BS Rater** pour labelliser des posts LinkedIn (slider 0–10), puis exporte ton dataset via le popup → *Exporter JSON*.

Un minimum de 100–200 posts labellisés est recommandé. Plus tu en as, plus le modèle sera précis.

---

## Entraîner le modèle

```bash
python train.py fichier1.json fichier2.json ...
```

Le script accepte un nombre arbitraire de fichiers JSON en entrée — utile pour fusionner plusieurs sessions de collecte.

Il affiche :
- La distribution de tes scores
- La MAE en validation croisée (Ridge vs SVR)
- Les mots les plus associés aux scores élevés (bullshit) et bas (clean)

Et produit deux fichiers :
- `tfidf_vocab.json` — à copier à la racine de l'extension
- `model.pkl` — pipeline complet pour inférence Python (non versionné)

---

## Mettre à jour l'extension

Une fois l'entraînement terminé, copie `tfidf_vocab.json` à la racine du repo :

```bash
cp tfidf_vocab.json ../tfidf_vocab.json
```

Recharge l'extension dans Chrome (`chrome://extensions` → bouton recharger) et le nouveau modèle est actif immédiatement.

---

## Interpréter les résultats

| Métrique | Description |
|---|---|
| MAE cross-val | Erreur moyenne sur données non vues — la métrique principale |
| MAE train | Erreur sur données d'entraînement — doit être plus basse que cross-val |

Une MAE cross-val autour de **2.2** est la baseline actuelle à 156 posts. Elle devrait descendre sous **2.0** avec 300+ posts.

Si MAE train << MAE cross-val (ex. 0.3 vs 2.2), le modèle overfitte — augmente `alpha` dans l'appel `Ridge(alpha=...)`. À l'inverse si MAE train ≈ MAE cross-val, tu peux essayer de baisser `alpha`.
