# 🛡️ LinkedIn Bullshit Detector (LBD)

A Chrome extension that helps you identify and manually rate low-quality LinkedIn posts — and collect labeled data to retrain the detection model on your own labels.

---

## What it does

LBD injects a small rating widget under every LinkedIn post in your feed. For each post, it shows:

- An **automatic score** predicted by a Ridge Regression model trained on labeled LinkedIn posts, running fully in-browser with no server required
- The **bullshit keywords** detected in the post text, shown as informational tags
- A **manual 0–10 slider** to record your own judgment
- A **save button** that stores the post text, author metadata, and both scores locally in your browser

The popup gives you a live overview of your dataset (total posts, average score, top offenders) and lets you export everything as a timestamped JSON file.

---

## Installation

No build step required. Load it directly as an unpacked extension:

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions`
3. Enable **Developer mode** (top right toggle)
4. Click **Load unpacked** and select the repository folder
5. Navigate to [linkedin.com](https://www.linkedin.com) — the widget will appear under each post

---

## How the auto-score works

The automatic score is predicted by a **Ridge Regression** model trained on manually labeled LinkedIn posts. It combines two types of features:

- **TF-IDF** — a weighted bag-of-words representation of the post text and author headline (500 features, uni- and bigrams)
- **Numeric features** — likes, comments, text length, word count, number of detected keywords, and whether the post was algorithmically promoted

The model weights live in `tfidf_vocab.json` at the root of the repo. Inference is done in pure JavaScript — no external dependencies, no network calls.

Keywords from `rules.js` are still displayed under the score as informational tags, but no longer drive the score itself.

Score thresholds: 🟢 < 4 / 🟠 4–6 / 🔴 ≥ 7.

---

## Retraining the model

The model can be retrained on your own labeled data. See [`training/README.md`](training/README.md) for instructions.

The short version:

```bash
cd training
pip install scikit-learn onnx skl2onnx
python train.py your_dataset.json
cp tfidf_vocab.json ../tfidf_vocab.json
```

Then reload the extension in Chrome — the new model is active immediately.

---

## Roadmap

- [x] Rule-based auto-scoring (heuristic baseline)
- [x] Manual rating widget + local storage
- [x] JSON export
- [x] Train a Ridge Regression model on labeled data (Python + scikit-learn)
- [x] In-browser inference — no server required
- [ ] Collect 300+ labeled posts and retrain for improved accuracy
- [ ] Collect ratings from other users to build a more consensual ground truth
- [ ] Explore alternative model architectures (e.g. gradient boosting, sentence embeddings, neural networks) as the dataset grows

---

## Data & privacy

All data stays **local in your browser** via `chrome.storage.local`. Nothing is sent anywhere. The JSON export is yours to use for training or analysis.

---

## License

MIT
