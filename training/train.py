#!/usr/bin/env python3
"""
LinkedIn Bullshit Detector — script d'entraînement
Usage : python train.py fichier1.json fichier2.json [...]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack
import unicodedata
import warnings
warnings.filterwarnings("ignore")

# Détection d'emojis : tout caractère dont la catégorie Unicode commence par "S" (symbol) ou "So"
def _is_emoji(c):
    cat = unicodedata.category(c)
    return cat in ("So", "Sm") or ord(c) > 0x1F000
EMOJI_CHARS = set(c for c in map(chr, range(0x110000)) if _is_emoji(c))

# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_datasets(paths):
    all_posts = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        posts = list(data.values()) if isinstance(data, dict) else data
        print(f"  {Path(path).name} : {len(posts)} posts")
        all_posts.extend(posts)
    return all_posts

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def parse_count(s):
    """'1 149 réactions' → 1149, '0' → 0"""
    if not s:
        return 0
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else 0

def feed_context_flag(fc):
    """'En fonction de votre profil...' → 1, tout le reste → 0"""
    if not fc:
        return 0
    return 1 if "profil" in fc.lower() else 0

def build_features(posts):
    texts       = []   # text + headline concaténés pour TF-IDF
    num_matrix  = []   # features numériques

    for p in posts:
        text     = p.get("text", "") or ""
        headline = p.get("headline", "") or ""
        combined = f"{text} {headline}".strip()
        texts.append(combined)

        likes      = parse_count(p.get("likes", ""))
        comments   = parse_count(p.get("comments", ""))
        auto_score = p.get("autoScore", 0) or 0
        text_len   = len(text)
        words      = text.split()
        word_count = len(words)
        fc_flag    = feed_context_flag(p.get("feedContext", ""))
        kw_count   = len(p.get("autoKeywords", []) or [])

        # Nouvelles features
        emoji_count  = sum(1 for c in text if c in EMOJI_CHARS)
        emoji_ratio  = emoji_count / max(word_count, 1)
        headline_len = len(headline)

        num_matrix.append([
            likes, comments, auto_score,
            text_len, word_count, fc_flag, kw_count,
            emoji_ratio, headline_len,
        ])

    labels = np.array([p.get("manualScore", 0) for p in posts], dtype=float)
    num_matrix = np.array(num_matrix, dtype=float)

    return texts, num_matrix, labels

# ============================================================
# 3. PIPELINE
# ============================================================

def build_pipeline(regressor):
    """
    TF-IDF (texte) + StandardScaler (numériques) → régression
    On utilise ColumnTransformer via une approche manuelle car
    TF-IDF attend une liste de strings, pas un array 2D.
    """
    # Stop words FR + EN combinés
    STOP_WORDS = list({
        # FR
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
        "à", "au", "aux", "ce", "se", "sa", "son", "ses", "mon", "ma",
        "mes", "ton", "ta", "tes", "leur", "leurs", "que", "qui", "quoi",
        "dont", "où", "ne", "pas", "plus", "par", "sur", "sous", "dans",
        "avec", "sans", "pour", "mais", "ou", "donc", "or", "ni", "car",
        "est", "sont", "était", "être", "avoir", "il", "elle", "ils",
        "elles", "nous", "vous", "je", "tu", "on", "y", "en", "si",
        "non", "oui", "très", "bien", "tout", "tous", "cette", "cet",
        "ces", "cela", "ça", "lui", "eux", "aussi", "même", "comme",
        "quand", "après", "avant", "entre", "lors", "puis",
        # EN
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "have", "has", "had", "do", "does", "did", "not",
        "this", "that", "these", "those", "it", "its", "we", "you",
        "he", "she", "they", "i", "my", "your", "our", "their", "his",
        "her", "as", "so", "if", "up", "out", "no", "all", "can", "will",
    })

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrammes + bigrammes
        max_features=500,
        sublinear_tf=True,    # log(tf) pour atténuer les répétitions
        min_df=4,             # ignore les mots apparaissant moins de 4 fois
        stop_words=STOP_WORDS,
    )
    scaler = StandardScaler()
    return tfidf, scaler, regressor

# ============================================================
# 4. EVALUATION
# ============================================================

def evaluate(name, regressor, texts, num_matrix, labels):
    tfidf, scaler, reg = build_pipeline(regressor)

    # Fit TF-IDF
    X_text = tfidf.fit_transform(texts)
    X_num  = scaler.fit_transform(num_matrix)

    from scipy.sparse import csr_matrix, hstack as sp_hstack
    X_num_sparse = csr_matrix(X_num)
    X = sp_hstack([X_text, X_num_sparse])

    # Cross-validation (5 folds, métrique MAE)
    scores = cross_val_score(reg, X, labels, cv=5, scoring="neg_mean_absolute_error")
    mae_mean = -scores.mean()
    mae_std  = scores.std()

    # Fit final sur tout le dataset
    reg.fit(X, labels)
    preds = reg.predict(X)
    mae_train = mean_absolute_error(labels, preds)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  MAE cross-val (5 folds) : {mae_mean:.2f} ± {mae_std:.2f}")
    print(f"  MAE train               : {mae_train:.2f}")

    return tfidf, scaler, reg, X

def print_top_words(tfidf, reg, n=15):
    """Affiche les mots les plus bullshit et les plus clean selon Ridge."""
    if not hasattr(reg, "coef_"):
        print("\n  (SVR : pas d'interprétabilité directe des coefficients)")
        return
    feature_names = tfidf.get_feature_names_out()
    coefs = reg.coef_[:len(feature_names)]
    top_bullshit = np.argsort(coefs)[-n:][::-1]
    top_clean    = np.argsort(coefs)[:n]

    print(f"\n  Top {n} mots → score ÉLEVÉ (bullshit) :")
    for i in top_bullshit:
        print(f"    +{coefs[i]:5.2f}  {feature_names[i]}")

    print(f"\n  Top {n} mots → score BAS (clean) :")
    for i in top_clean:
        print(f"    {coefs[i]:5.2f}  {feature_names[i]}")

# ============================================================
# 5. EXPORT ONNX
# ============================================================

def export_onnx(tfidf, scaler, reg, texts, num_matrix, labels, output_path="model.onnx"):
    """
    Export du modèle complet en deux artefacts :
    1. model.pkl  — pipeline complet (TF-IDF + scaler + Ridge) pour inférence Python
    2. model.onnx — Ridge sur features combinées (TF-IDF float32 + numériques)
                    pour inférence in-browser via ONNX Runtime Web
    """
    import pickle
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # — Pickle : pipeline complet —
    artifacts = {"tfidf": tfidf, "scaler": scaler, "regressor": reg}
    pkl_path = output_path.replace(".onnx", ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n✅ Pipeline complet sauvegardé : {pkl_path}")

    # — ONNX : Ridge re-fitté sur features combinées float32 —
    X_text = tfidf.transform(texts).toarray().astype(np.float32)
    X_num  = scaler.transform(num_matrix).astype(np.float32)
    X_combined = np.hstack([X_text, X_num])

    ridge_onnx = Ridge(alpha=reg.alpha if hasattr(reg, "alpha") else 0.1)
    ridge_onnx.fit(X_combined, labels)

    n_features = X_combined.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(ridge_onnx, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Modèle ONNX sauvegardé    : {output_path}")
    print(f"   ({n_features} features = {X_text.shape[1]} TF-IDF + {X_num.shape[1]} numériques)")

# ============================================================
# 6. MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage : python train.py fichier1.json fichier2.json [...]")
        sys.exit(1)

    paths = sys.argv[1:]
    print(f"\n📂 Chargement de {len(paths)} fichier(s)...")
    posts = load_datasets(paths)
    print(f"   Total : {len(posts)} posts\n")

    # Filtrer les posts sans manualScore
    posts = [p for p in posts if p.get("manualScore") is not None]
    print(f"   Posts avec manualScore : {len(posts)}")

    texts, num_matrix, labels = build_features(posts)

    print(f"\n📊 Distribution des scores :")
    from collections import Counter
    dist = Counter(int(l) for l in labels)
    for score in range(11):
        bar = "█" * dist.get(score, 0)
        print(f"  {score:2d} │{bar} {dist.get(score, 0)}")

    # Évaluation Ridge
    tfidf_r, scaler_r, ridge, X_r = evaluate(
        "Ridge Regression (alpha=0.1)",
        Ridge(alpha=0.1),
        texts, num_matrix, labels
    )
    print_top_words(tfidf_r, ridge)

    # Évaluation SVR
    tfidf_s, scaler_s, svr, X_s = evaluate(
        "SVR (kernel=rbf)",
        SVR(kernel="rbf", C=1.0, epsilon=0.5),
        texts, num_matrix, labels
    )

    # Choix du meilleur modèle
    print(f"\n{'═'*50}")
    print("  → Le modèle Ridge est recommandé pour l'export")
    print("     (interprétable + performant sur petits datasets)")
    print(f"{'═'*50}")

    # Export
    export_onnx(tfidf_r, scaler_r, ridge, texts, num_matrix, labels)

if __name__ == "__main__":
    main()
