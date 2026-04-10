# Training — LinkedIn Bullshit Detector

This folder contains the bootstrap training script for the automatic scoring model.

Under normal use, **you don't need to run this**. The extension retrains the model directly in your browser from the popup, with no Python required. This script exists for reproducibility — it lets you regenerate `tfidf_vocab.json` from scratch if you want to change the vocabulary, tweak hyperparameters, or understand exactly how the base model was built.

---

## What this script produces

The model is a **Ridge Regression** trained on two types of features:

- **TF-IDF** — bag-of-words representation of the post text and author headline (500 features, uni- and bigrams)
- **Numeric features** — likes, comments, text length, word count, emoji ratio, keyword count, whether the post was promoted

The output is `tfidf_vocab.json` — a ~34 KB JSON file containing the vocabulary, IDF weights, Ridge coefficients, and scaler parameters. This file is the base model loaded by the extension when no personalized model exists yet.

Current hyperparameters: `alpha=0.1`, `min_df=4`, `max_features=500`, bigrams enabled.

---

## Requirements

Python 3.9+ and pip:

```bash
pip install scikit-learn skl2onnx
```

---

## Usage

```bash
python train.py file1.json file2.json ...
```

The script accepts any number of JSON files — useful for merging multiple labeling sessions. Each file should be an export from the extension's Collect mode (popup → *Export JSON*).

It prints:
- The distribution of your scores
- Cross-validated MAE (5 folds) for Ridge and SVR
- The words most associated with high (bullshit) and low (clean) scores

And writes two files:
- `tfidf_vocab.json` — copy this to the repo root to update the base model
- `model.pkl` — full pipeline for Python inference (not versioned)

---

## Updating the extension

After training, copy `tfidf_vocab.json` to the repo root:

```bash
cp tfidf_vocab.json ../tfidf_vocab.json
```

Reload the extension in Chrome (`chrome://extensions` → reload button). The new base model is active immediately.

Note: any personalized model trained via the popup takes priority over this file. To fall back to the base model, use the *Revert to base model* button in the popup.

---

## Interpreting results

| Metric | Description |
|---|---|
| Cross-val MAE | Average error on unseen data (5-fold) — the primary metric |
| Train MAE | Error on training data — should be lower than cross-val MAE |

A cross-val MAE around **2.2** is the current baseline at 156 posts. It should drop below **2.0** with 300+ posts.

If train MAE << cross-val MAE (e.g. 0.3 vs 2.2), the model is overfitting — increase `alpha` in the `Ridge(alpha=...)` call. If train MAE ≈ cross-val MAE, you can try lowering `alpha`.

In-browser retraining uses an adaptive alpha (`20 / n_samples`) so you don't need to tune this manually there.
