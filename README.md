# 🛡️ LinkedIn Bullshit Detector (LBD)

A Chrome extension that helps you identify and manually rate low-quality LinkedIn posts — and collect labeled data to eventually train a real detection model.

---

## What it does

LBD injects a small rating widget under every LinkedIn post in your feed. For each post, it shows:

- An **automatic pre-score** based on keyword detection and author headline analysis (bullshit jargon, personal pronouns, suspiciously long titles…)
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

The automatic score is a heuristic starting point, not a ground truth. It combines two signals:

- **Keyword score** — counts buzzwords and bullshit patterns in the post text (e.g. *"thought leader"*, *"excited to announce"*, *"resilience"*), scaled logarithmically
- **Headline penalty** — flags author titles containing personal pronouns, marketing jargon (™, *ninja*, *solopreneur*…), or suspiciously formatted service lists

Final score is capped at 10. Thresholds: 🟢 < 4 / 🟠 4–6 / 🔴 ≥ 7.

All keyword rules live in `rules.js` and are easy to extend.

---

## Roadmap

This extension is the **data collection phase** of a larger project:

- [x] Rule-based auto-scoring (heuristic baseline)
- [x] Manual rating widget + local storage
- [x] JSON export
- [ ] Collect ~200–300 labeled posts
- [ ] Train a lightweight classifier on the exported data (Python + scikit-learn / PyTorch)
- [ ] Convert the model to run fully in-browser (TensorFlow.js or ONNX Runtime Web)
- [ ] Replace the heuristic auto-score with the trained model — no server required

---

## Data & privacy

All data stays **local in your browser** via `chrome.storage.local`. Nothing is sent anywhere. The JSON export is yours to use for training or analysis.

---

## License

MIT
