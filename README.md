# 🛡️ LinkedIn Bullshit Detector (LBD)

A Chrome extension that detects and filters low-quality LinkedIn posts using a Ridge Regression model that can be retrained on your own labels in one click. Everything run on your browser.

---

## What it does

LBD runs in two modes, switchable from the popup:

**🚫 Filter mode** : automatically hides posts above a configurable bullshit threshold. Sponsored posts can also be hidden. A placeholder shows the score and lets you reveal the post if you want.

**🧪 Collect mode** : injects a rating widget under every post. For each post it shows the automatic score and a 0–10 slider to record your own judgment. Labeled posts are saved locally only.

Once you've labeled enough posts, you can retrain the model directly from the popup in one click.

---

## Installation

No build step required. Load it directly as an unpacked extension:

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions`
3. Enable **Developer mode** (top right toggle)
4. Click **Load unpacked** and select the repository folder
5. Navigate to [linkedin.com](https://www.linkedin.com)

---

## How the model works

Posts are scored by a **Ridge Regression** model combining two types of features:

- **TF-IDF** : a weighted bag-of-words representation of the post text and author headline (500 features)
- **Numeric features** : likes, comments, text length, word count, emoji ratio, keyword count, phrase lengths

Inference runs in pure JavaScript. No external dependencies, no network calls.

---

## Retraining the model

### In-browser (recommended)

Once you've labeled posts in Collect mode, click **🔁 Retrain** in the popup. The extension will:

1. Recompute TF-IDF weights (IDF) from your labeled corpus
2. Refit the StandardScaler on your numeric features
3. Solve Ridge analytically
4. Report a validation MAE on a held-out 20% split (deterministic, stable across runs)
5. Retrain on 100% of your data and save the model to `chrome.storage.local`

The regularization strength (alpha) is chosen automatically based on dataset size. The model activates immediately, no reload needed.

At least 20 labeled posts are required to retrain; 50+ are recommended for reliable results.

### From scratch with Python (for reproducibility)

`train.py` generates the bootstrap `tfidf_vocab.json` the fixed vocabulary, initial IDF weights, and baseline Ridge coefficients. It is kept in the repo so the full pipeline can be reproduced from scratch.

```bash
pip install scikit-learn skl2onnx
python train.py your_dataset.json
```

This overwrites `tfidf_vocab.json`. Reload the extension in Chrome to apply. Under normal use you don't need to run this, the in-browser retraining handles everything.

---

## Data & privacy

All data stays **local in your browser** via `chrome.storage.local`. Nothing is sent anywhere. The JSON export is yours to use for training or analysis.

---

## License

MIT
