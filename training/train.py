#!/usr/bin/env python3
"""
LinkedIn Bullshit Detector — script d'entraînement
Usage : python train.py fichier1.json fichier2.json [...]

Génère tfidf_vocab.json — à placer dans le dossier de l'extension.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix, hstack
import unicodedata
import warnings
warnings.filterwarnings("ignore")

# ── Détection d'emojis ──
def _is_emoji(c):
    cat = unicodedata.category(c)
    return cat in ("So", "Sm") or ord(c) > 0x1F000
EMOJI_CHARS = set(c for c in map(chr, range(0x110000)) if _is_emoji(c))

# ── Stop words FR + EN ──
STOP_WORDS = list({
    "le","la","les","de","du","des","un","une","et","en","à","au","aux",
    "ce","se","sa","son","ses","mon","ma","mes","ton","ta","tes","leur",
    "leurs","que","qui","quoi","dont","où","ne","pas","plus","par","sur",
    "sous","dans","avec","sans","pour","mais","ou","donc","or","ni","car",
    "est","sont","était","être","avoir","il","elle","ils","elles","nous",
    "vous","je","tu","on","y","si","non","oui","très","bien","tout","tous",
    "cette","cet","ces","cela","ça","lui","eux","aussi","même","comme",
    "quand","après","avant","entre","lors","puis",
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had",
    "do","does","did","not","this","that","these","those","it","its","we",
    "you","he","she","they","i","my","your","our","their","his","her","as",
    "so","if","up","out","no","all","can","will",
})

NUM_FEATURE_NAMES = [
    "likes", "comments", "text_len",
    "word_count", "fc_flag", "emoji_ratio", "headline_len",
]

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
    if not s:
        return 0
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else 0

def feed_context_flag(fc):
    return 1 if fc and "profil" in fc.lower() else 0

def build_features(posts):
    texts      = []
    num_matrix = []

    for p in posts:
        text     = p.get("text", "") or ""
        headline = p.get("headline", "") or ""
        texts.append(f"{text} {headline}".strip())

        words        = text.split()
        word_count   = len(words)
        emoji_count  = sum(1 for c in text if c in EMOJI_CHARS)
        emoji_ratio  = emoji_count / max(word_count, 1)

        num_matrix.append([
            parse_count(p.get("likes", "")),
            parse_count(p.get("comments", "")),
            len(text),
            word_count,
            feed_context_flag(p.get("feedContext", "")),
            emoji_ratio,
            len(headline),
        ])

    labels     = np.array([p.get("manualScore", 0) for p in posts], dtype=float)
    num_matrix = np.array(num_matrix, dtype=float)
    return texts, num_matrix, labels

# ============================================================
# 3. ENTRAÎNEMENT
# ============================================================

def train(texts, num_matrix, labels, alpha=0.1, min_df=3):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
        min_df=min_df,
        stop_words=STOP_WORDS,
    )
    X_text = tfidf.fit_transform(texts)

    scaler = StandardScaler()
    X_num  = scaler.fit_transform(num_matrix)
    X      = hstack([X_text, csr_matrix(X_num)])

    ridge = Ridge(alpha=alpha)

    # Cross-validation
    cv_scores = cross_val_score(ridge, X, labels, cv=5, scoring="neg_mean_absolute_error")
    mae_cv  = -cv_scores.mean()
    mae_std = cv_scores.std()
    print(f"  MAE cross-val (5 folds) : {mae_cv:.2f} ± {mae_std:.2f}")

    # Fit final
    ridge.fit(X, labels)
    preds     = np.clip(ridge.predict(X), 0, 10)
    mae_train = mean_absolute_error(labels, preds)
    print(f"  MAE train               : {mae_train:.2f}")

    return tfidf, scaler, ridge

# ============================================================
# 4. INTERPRÉTABILITÉ
# ============================================================

def print_top_words(tfidf, ridge, n=15):
    feature_names = tfidf.get_feature_names_out()
    coefs = ridge.coef_[:len(feature_names)]

    print(f"\n  Top {n} tokens → score ÉLEVÉ (bullshit) :")
    for i in np.argsort(coefs)[-n:][::-1]:
        print(f"    +{coefs[i]:5.2f}  {feature_names[i]}")

    print(f"\n  Top {n} tokens → score BAS (clean) :")
    for i in np.argsort(coefs)[:n]:
        print(f"    {coefs[i]:5.2f}  {feature_names[i]}")

    print(f"\n  Features numériques :")
    num_coefs = ridge.coef_[len(feature_names):]
    for name, coef in zip(NUM_FEATURE_NAMES, num_coefs):
        print(f"    {coef:+.3f}  {name}")

# ============================================================
# 5. EXPORT tfidf_vocab.json
# ============================================================

def export_vocab(tfidf, scaler, ridge, output_path="tfidf_vocab.json"):
    vocab_list = [None] * len(tfidf.vocabulary_)
    for w, i in tfidf.vocabulary_.items():
        vocab_list[i] = w

    output = {
        "vocabulary":        {w: int(i) for w, i in tfidf.vocabulary_.items()},
        "idf":               tfidf.idf_.tolist(),
        "vocab_list":        vocab_list,
        "sublinear_tf":      True,
        "scaler_mean":       scaler.mean_.tolist(),
        "scaler_scale":      scaler.scale_.tolist(),
        "ridge_coef":        ridge.coef_.tolist(),
        "ridge_intercept":   float(ridge.intercept_),
        "n_tfidf_features":  int(len(tfidf.vocabulary_)),
        "n_num_features":    len(NUM_FEATURE_NAMES),
        "num_feature_names": NUM_FEATURE_NAMES,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    n_tfidf = len(tfidf.vocabulary_)
    n_num   = len(NUM_FEATURE_NAMES)
    print(f"\n✅ {output_path} exporté ({n_tfidf} TF-IDF + {n_num} numériques = {n_tfidf + n_num} features)")

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
    print(f"   Total : {len(posts)} posts")

    posts = [p for p in posts if p.get("manualScore") is not None]
    print(f"   Labellisés : {len(posts)}\n")

    texts, num_matrix, labels = build_features(posts)

    print("📊 Distribution des scores :")
    dist = Counter(int(l) for l in labels)
    for score in range(11):
        bar = "█" * dist.get(score, 0)
        print(f"  {score:2d} │{bar} {dist.get(score, 0)}")

    print("\n🔧 Entraînement Ridge (alpha=0.1) :")
    tfidf, scaler, ridge = train(texts, num_matrix, labels, alpha=0.1, min_df=3)

    print_top_words(tfidf, ridge)
    export_vocab(tfidf, scaler, ridge)

if __name__ == "__main__":
    main()
