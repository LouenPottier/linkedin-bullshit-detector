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

NUM_FEATURE_NAMES_BASE = [
    "likes", "comments", "text_len",
    "word_count", "emoji_ratio", "headline_len",
    "short_sent_3", "short_sent_5", "short_sent_7", "short_sent_10",
]
TOP_N_EMOJIS = 10

# ============================================================
# SIGMOID / LOGIT
# ============================================================

def logit10(y):
    p = np.clip(y / 10.0, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid10(x, k=1.0):
    return 10.0 / (1.0 + np.exp(-k * x))

# Hash déterministe identique au JS (djb2 signed int32)
def hash_post_id(post_id):
    h = 5381
    for c in (post_id or ""):
        h = ((h << 5) + h) ^ ord(c)
        h = h & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    return abs(h)

# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_datasets(paths):
    all_posts = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
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

def count_short_sentences(text, max_words):
    """Nombre de phrases de moins de max_words mots."""
    if not text:
        return 0
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
    return sum(1 for s in sentences if len(s.split()) < max_words)

def extract_top_emojis(texts, top_n=TOP_N_EMOJIS):
    """Retourne les top_n emojis les plus fréquents (présence par document)."""
    counts = Counter()
    for text in texts:
        seen = set()
        for c in text:
            if c in EMOJI_CHARS and c not in seen:
                counts[c] += 1
                seen.add(c)
    return [em for em, _ in counts.most_common(top_n)]

def build_features(posts, top_emojis):
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

        # Base (6)
        base = [
            parse_count(p.get("likes", "")),
            parse_count(p.get("comments", "")),
            len(text),
            word_count,
            emoji_ratio,
            len(headline),
        ]

        # Phrases courtes (4)
        short_sent = [count_short_sentences(text, k) for k in (3, 5, 7, 10)]

        # Top emojis (15)
        emoji_feats = [text.count(em) for em in top_emojis]

        num_matrix.append(base + short_sent + emoji_feats)

    labels     = np.array([p.get("manualScore", 0) for p in posts], dtype=float)
    num_matrix = np.array(num_matrix, dtype=float)
    return texts, num_matrix, labels

# ============================================================
# 3. ENTRAÎNEMENT
# ============================================================

def train(texts, num_matrix, labels, post_ids, alpha=0.5, min_df=3, slope=0.5):
    # Split déterministe : hash(postId) % 5 == 0 → val
    val_mask   = np.array([hash_post_id(pid) % 5 == 0 for pid in post_ids])
    train_mask = ~val_mask
    if val_mask.sum() == 0:
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[::5] = True
        train_mask = ~val_mask

    y_logit = logit10(labels)

    # ── Modèle val : fitté uniquement sur train, sans leak ──
    def make_Xy(mask):
        tfidf_v = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                                  sublinear_tf=True, min_df=min_df, stop_words=STOP_WORDS)
        X_text = tfidf_v.fit_transform([texts[i] for i in np.where(mask)[0]])
        scaler_v = StandardScaler()
        X_num = scaler_v.fit_transform(num_matrix[mask])
        X = hstack([X_text, csr_matrix(X_num)]).toarray()
        return tfidf_v, scaler_v, X

    tfidf_tr, scaler_tr, X_train = make_Xy(train_mask)

    # Transformer val avec le vocab/scaler du train (sans leak)
    X_val_text = tfidf_tr.transform([texts[i] for i in np.where(val_mask)[0]])
    X_val_num  = scaler_tr.transform(num_matrix[val_mask])
    X_val      = hstack([X_val_text, csr_matrix(X_val_num)]).toarray()

    ridge_val = Ridge(alpha=alpha)
    ridge_val.fit(X_train, y_logit[train_mask])
    preds_val = np.clip(sigmoid10(ridge_val.predict(X_val), slope), 0, 10)
    mae_val   = mean_absolute_error(labels[val_mask], preds_val)
    print(f"  MAE val  ({val_mask.sum()} posts, sans leak) : {mae_val:.2f}")

    # ── Modèle full : fitté sur 100% des données ──
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                            sublinear_tf=True, min_df=min_df, stop_words=STOP_WORDS)
    X_text_full = tfidf.fit_transform(texts)
    scaler = StandardScaler()
    X_num_full  = scaler.fit_transform(num_matrix)
    X_full      = hstack([X_text_full, csr_matrix(X_num_full)]).toarray()

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_full, y_logit)
    preds_train = np.clip(sigmoid10(ridge.predict(X_full), slope), 0, 10)
    mae_train   = mean_absolute_error(labels, preds_train)
    print(f"  MAE train ({len(labels)} posts) : {mae_train:.2f}")

    return tfidf, scaler, ridge

# ============================================================
# 4. INTERPRÉTABILITÉ
# ============================================================

def print_top_words(tfidf, ridge, num_feature_names, n=15):
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
    for name, coef in zip(num_feature_names, num_coefs):
        print(f"    {coef:+.3f}  {name}")

# ============================================================
# 5. EXPORT tfidf_vocab.json
# ============================================================

def export_vocab(tfidf, scaler, ridge, top_emojis, slope, output_path="tfidf_vocab.json"):
    vocab_list = [None] * len(tfidf.vocabulary_)
    for w, i in tfidf.vocabulary_.items():
        vocab_list[i] = w

    num_feature_names = (
        NUM_FEATURE_NAMES_BASE +
        [f"emoji_{'_'.join(hex(ord(c))[2:] for c in em)}" for em in top_emojis]
    )

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
        "n_num_features":    len(num_feature_names),
        "num_feature_names": num_feature_names,
        "top_emojis":        top_emojis,
        "use_sigmoid":       True,
        "sigmoid_slope":     slope,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    n_tfidf = len(tfidf.vocabulary_)
    n_num   = len(num_feature_names)
    print(f"\n✅ {output_path} exporté ({n_tfidf} TF-IDF + {n_num} numériques = {n_tfidf + n_num} features)")

# ============================================================
# 6. MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LinkedIn Bullshit Detector — entraînement")
    parser.add_argument("files", nargs="+", help="Fichiers JSON du dataset")
    parser.add_argument("--alpha", type=float, default=0.5, help="Régularisation Ridge (défaut: 0.5)")
    parser.add_argument("--slope", type=float, default=0.5, help="Pente de la sigmoid (défaut: 0.5)")
    args = parser.parse_args()

    paths = args.files
    alpha = args.alpha
    slope = args.slope
    print(f"\n📂 Chargement de {len(paths)} fichier(s)...")
    posts = load_datasets(paths)
    print(f"   Total : {len(posts)} posts")

    posts = [p for p in posts if p.get("manualScore") is not None]
    print(f"   Labellisés : {len(posts)}\n")

    # Calculer les top emojis sur le corpus entier
    all_texts = [p.get("text", "") or "" for p in posts]
    top_emojis = extract_top_emojis(all_texts, TOP_N_EMOJIS)
    print(f"🔤 Top {len(top_emojis)} emojis : {' '.join(top_emojis)}\n")

    post_ids = [p.get("postId", "") or "" for p in posts]
    texts, num_matrix, labels = build_features(posts, top_emojis)

    print("📊 Distribution des scores :")
    dist = Counter(int(l) for l in labels)
    for score in range(11):
        bar = "█" * dist.get(score, 0)
        print(f"  {score:2d} │{bar} {dist.get(score, 0)}")

    print(f"\n🔧 Entraînement Ridge (alpha={alpha}) :")
    tfidf, scaler, ridge = train(texts, num_matrix, labels, post_ids, alpha=alpha, min_df=3, slope=slope)

    num_feature_names = (
        NUM_FEATURE_NAMES_BASE +
        [f"emoji_{'_'.join(hex(ord(c))[2:] for c in em)}" for em in top_emojis]
    )
    print_top_words(tfidf, ridge, num_feature_names)
    export_vocab(tfidf, scaler, ridge, top_emojis, slope)

if __name__ == "__main__":
    main()
