# HAM10000 Skin Lesion Classifier

An educational, end-to-end machine learning project that classifies dermatoscopic
images of skin lesions into 7 diagnostic categories using the **HAM10000** dataset.

The project deliberately trains **two model families** so you can compare a classic
machine-learning approach against modern deep learning:

| Option | Approach | What it teaches |
|--------|----------|-----------------|
| **A** | EfficientNet-B0 (frozen) → embeddings → classic head (Logistic Regression by default; SVM / XGBoost optional) | Feature extraction + the "linear probe" idea |
| **B** | EfficientNet-B0 **fine-tuned** end-to-end | Transfer learning by fine-tuning |

Comparing A vs B is the textbook **"linear probing vs fine-tuning"** experiment.

> ⚠️ **Disclaimer — not a medical device.** This project is for learning and
> experimentation only. It is **not** validated for clinical use and must never be
> used to make real diagnostic or treatment decisions. Always consult a qualified
> dermatologist.

---

## The dataset

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
("Human Against Machine with 10000 training images") contains ~10,015 dermatoscopic
images across 7 classes:

| Code | Meaning |
|------|---------|
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `bkl` | Benign keratosis-like lesions |
| `bcc` | Basal cell carcinoma |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `vasc` | Vascular lesions |
| `df` | Dermatofibroma |

Two characteristics drive the whole design:

1. **Severe class imbalance** — `nv` is ~67% of the data; `df` and `vasc` are tiny.
   We therefore report **balanced accuracy, macro-F1, and per-class recall** (not raw
   accuracy) and use class weighting during training.
2. **Multiple images per lesion** — the metadata groups images by `lesion_id`. We
   **split by `lesion_id`** so images of the same lesion never leak across
   train/val/test.

The dataset is **not** committed to this repo. See [`data/README.md`](data/README.md)
for the download steps.

---

## Project structure

```
ham10000-skin-lesion-classifier/
├── README.md
├── INSTRUCTIONS.md          # Step-by-step build guide (the project roadmap)
├── requirements.txt
├── .gitignore
├── config.yaml              # Paths, hyperparameters, class list
├── data/
│   └── README.md            # How to download HAM10000
├── src/
│   ├── data.py              # Loading, lesion-id split, datasets, transforms
│   ├── features.py          # Frozen EfficientNet-B0 embedding extraction (Option A)
│   ├── train_classic.py     # Train LR / SVM / XGBoost on cached embeddings (Option A)
│   ├── train_cnn.py         # Fine-tune EfficientNet-B0 (Option B)
│   ├── evaluate.py          # Shared metrics + confusion matrix
│   ├── predict.py           # Shared inference: predict(image) -> {label, probs}
│   └── utils.py
├── api/
│   └── main.py              # FastAPI app (imports src/predict.py)
├── app/
│   └── app.py               # Gradio app for the Hugging Face Space
├── models/                  # Saved artifacts (gitignored; tracked via MLflow)
├── notebooks/               # Optional EDA
└── tests/
    └── test_predict.py
```

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the data (see data/README.md), then build the train/val/test split
python -m src.data --build-split

# 3. Option A — extract embeddings once, train classic heads (fast, CPU-friendly)
python -m src.features        # caches embeddings to models/embeddings.npz
python -m src.train_classic --head logreg   # also: svm, xgboost

# 4. Option B — fine-tune EfficientNet-B0 (uses GPU if available)
python -m src.train_cnn

# 5. Compare runs in MLflow
mlflow ui                     # open http://127.0.0.1:5000
```

### Run the API locally

```bash
uvicorn api.main:app --reload
# open http://127.0.0.1:8000/docs  and try POST /predict with an image
```

### Run the demo website locally

```bash
python app/app.py             # opens a Gradio UI: upload an image, get the prediction
```

---

## Tracking & metrics (MLflow)

Every training run logs:

- **Params:** model option, head type, learning rate, epochs, augmentation settings, seed.
- **Metrics:** accuracy, **balanced accuracy**, **macro-F1**, weighted-F1, **per-class recall**.
- **Artifacts:** confusion matrix plot, the fitted model, the label mapping.

This lets Option A and Option B sit side by side in the MLflow UI for a fair comparison.

---

## Deployment (Hugging Face)

- **Model Hub** — model weights + a model card (including the disclaimer above).
- **Space (Gradio)** — the "upload an image → get the detection" website. The Space
  reuses `src/predict.py`, so the demo and the API share identical inference logic.

---

## Tech stack

PyTorch · timm · scikit-learn · XGBoost · MLflow · FastAPI · Gradio · pandas · Pillow

---

## License & data use

Code: choose a license (MIT is common for learning projects). The HAM10000 dataset has
its own license/terms (CC BY-NC) — review them before redistributing images or weights.
