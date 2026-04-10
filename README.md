# 🛡️ LinkedIn Bullshit Detector (LBD)

A Chrome extension that hides low-quality LinkedIn posts automatically.

Install it, browse LinkedIn — it works straight away. The extension ships with a pretrained model that scores every post in your feed and hides the ones above a threshold you control with a slider. Calibrate your bullshit tolerance, done. No account, no setup, no code required.

If you want it to match your taste rather than mine, you can rate posts yourself and retrain the model on your labels. That part requires running a Python script, but it's optional — the default model is usable on its own.

Everything stays local. Nothing is sent anywhere.

---

## How it works

### 1. Collect mode — label your feed

Switch to collect mode in the popup. A rating widget appears under every LinkedIn post, showing:

- An **automatic score** from the current model (0–10)
- The **bullshit keywords** detected in the post, as informational tags
- A **manual slider** (0–10) to record your own judgment
- A **save button** that stores the post text, author metadata, and both scores in your browser

Once you've rated enough posts (~100 is a good start, 300+ gives better results), export your dataset as a JSON file from the popup.

### 2. Train — personalize the model

Run the training script on your exported data:

```bash
pip install scikit-learn onnx skl2onnx
python training/train.py your_dataset.json
cp tfidf_vocab.json path/to/extension/tfidf_vocab.json
```

This trains a Ridge Regression model on your labels, evaluates it against an SVR baseline, and prints the top words driving the score up or down (useful for sanity-checking your labels). Model weights are written to `tfidf_vocab.json`. Copy it to the extension root, reload the extension in Chrome — the new model is live immediately.

### 3. Filter mode — hide the bullshit

Switch to filter mode. Posts whose score exceeds the threshold (adjustable in the popup, default 7/10) are automatically hidden behind a dismissible placeholder. The threshold and sensitivity are yours to tune.

---

## Installation

No build step required.

1. Clone or download this repository
2. Open Chrome and go to `chrome://extensions`
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked** and select the repository folder
5. Go to [linkedin.com](https://www.linkedin.com) — the extension activates automatically

---

## How the model works

The automatic score is predicted by a **Ridge Regression** model running fully in-browser (pure JavaScript, no dependencies, no network calls). It combines:

- **TF-IDF** — a weighted bag-of-words of the post text and author headline (500 features, uni- and bigrams)
- **Numeric features** — likes, comments, text length, word count, emoji ratio, keyword count, headline length, and whether the post was algorithmically promoted

Model weights live in `tfidf_vocab.json`. The default model shipped in the repo was trained on a small initial dataset — it works as a baseline, but it gets significantly better once retrained on your own labels.

Keywords from `rules.js` are shown under each post as informational tags but no longer drive the score.

Score thresholds: 🟢 < 4 / 🟠 4–6 / 🔴 ≥ 7.

---

## Roadmap

- [x] Rule-based auto-scoring (heuristic baseline)
- [x] Manual rating widget + local storage
- [x] JSON export
- [x] Dual-mode UI — collect and filter in the same extension
- [x] Ridge Regression model trained on personal labeled data
- [x] In-browser inference — no server required
- [x] Adjustable filter threshold in the popup
- [ ] Collect 300+ labeled posts and retrain for improved accuracy
- [ ] Explore alternative architectures (gradient boosting, sentence embeddings) as the dataset grows

---

## Data & privacy

All data stays **local** in `chrome.storage.local`. Nothing is transmitted. The JSON export is yours — use it to retrain, analyse, or share the dataset if you choose to.

---

## License

MIT
