# Cricket Shot Attention CNN

Fine-grained classification of cricket batting shots from **single frames** — pretrained CNNs, a custom CNN trained from scratch, and a CBAM attention variant, evaluated under a leakage-controlled protocol.

![status](https://img.shields.io/badge/status-in%20progress-orange)
![submission](https://img.shields.io/badge/submission-1%20of%204-blue)
![license](https://img.shields.io/badge/code-MIT-green)

---

## Problem

Given one still frame of a batsman mid-stroke, predict which of four batting shots is being played:

| Class | Description |
|---|---|
| `drive` | Cover drive, straight drive, off drive |
| `legglance-flick` | Leg glance and flick |
| `pullshot` | Pull |
| `sweep` | Sweep |

**Why this is hard.** All four classes share the same scene grammar — one batsman, one pitch, the same kit, similar stadium backdrops. The discriminative signal is confined to body posture and bat angle, which occupy a small fraction of the frame. The model must learn to ignore the crowd, the sightscreen and the advertising boards.

**Why single frames.** Almost all prior cricket shot work is video-based and relies on temporal models (LSTM/GRU over frame sequences). This project asks how much of the task survives when that temporal signal is removed, and whether spatial attention can recover part of the gap.

---

## Dataset

| | |
|---|---|
| **Name** | Cricket Shot Dataset |
| **Source** | https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset |
| **Task** | Multi-class, single-label image classification |
| **Classes** | 4 |
| **Images** | *computed in Submission 1 — see notebook output* |
| **Licence** | *verify on the Kaggle page before redistributing* |

> **Note on construction.** This dataset was assembled from internet images and then expanded by offline augmentation. Multiple files are therefore transformations of the same source photograph. See [Known issue: duplicate leakage](#known-issue-duplicate-leakage).

**The dataset is not committed to this repository.** Download it from Kaggle, or add it as an input on a Kaggle notebook where it mounts at `/kaggle/input/cricket-shot-dataset/data`.

---

## Known issue: duplicate leakage

Because the published dataset contains offline-augmented copies of the same source images, a naive `train_test_split` places transformations of one photograph into both train and test. The model is then evaluated on pictures it has effectively already seen, and every reported metric is inflated.

This repository controls for it:

1. A 64-bit perceptual difference hash (dHash) is computed for every image.
2. Images sharing a hash are collected into **duplicate groups**.
3. The train/val/test split is made over *groups*, stratified by class — never over individual files.
4. The resulting split is frozen to [`artifacts/split_manifest.csv`](artifacts/) and reused unchanged by every later submission.

This matters beyond tidiness. A 2025 standardized re-implementation study found that cricket shot models originally reporting 96%, 99.2% and 93% accuracy scored 46.0%, 55.6% and 57.7% when re-run under a consistent protocol ([Kang 2025](https://arxiv.org/abs/2510.09187)). Uncontrolled splits are a plausible contributor.

---

## Submissions

| # | Deliverable | Notebook | Status |
|---|---|---|---|
| 1 | Dataset, EDA, related work | [`01_eda_and_related_work.ipynb`](notebooks/) | 🟡 In progress |
| 2 | 3+ pretrained models, frozen vs fine-tuned | `02_pretrained_models.ipynb` | ⚪ Not started |
| 3 | Custom CNN from scratch | `03_custom_cnn.ipynb` | ⚪ Not started |
| 4 | CNN + CBAM attention | `04_cnn_with_attention.ipynb` | ⚪ Not started |

From Submission 2 onward, every notebook reports accuracy, precision, recall, F1, a confusion matrix, and training/testing time.

---

## Results

*Populated as submissions complete.*

| Model | Mode | Accuracy | Macro-F1 | Train time | Notes |
|---|---|---|---|---|---|
| ResNet50 | frozen | — | — | — | |
| ResNet50 | fine-tuned | — | — | — | |
| EfficientNet-B0 | frozen | — | — | — | |
| EfficientNet-B0 | fine-tuned | — | — | — | |
| DenseNet121 | frozen | — | — | — | |
| DenseNet121 | fine-tuned | — | — | — | |
| Custom CNN | scratch | — | — | — | Submission 3 baseline |
| Custom CNN + CBAM | scratch | — | — | — | Submission 4 |

---

## Repository layout

```
cricket-shot-attention-cnn/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda_and_related_work.ipynb
│   ├── 02_pretrained_models.ipynb   # planned
│   ├── 03_custom_cnn.ipynb          # planned
│   └── 04_cnn_with_attention.ipynb  # planned
├── artifacts/
│   └── split_manifest.csv        # frozen split — written by notebook 01, do not regenerate
├── reports/
│   ├── Submission1_Report.pdf
│   ├── Submission1_Report.tex
│   └── figures/                  # sample frames used in the report
└── results/
    └── figures/
```

---

## Reproducing

```bash
git clone https://github.com/Rayhan-who/cricket-shot-attention-cnn
cd cricket-shot-attention-cnn
```

Submission 1 needs only `numpy`, `pandas`, `matplotlib`, `pillow` and `scikit-learn` — all preinstalled on Kaggle and Colab. Locally: `pip install -r requirements.txt`.

Then: add the dataset as a Kaggle input (or set `DATA_DIR`), open `notebooks/01_eda_and_related_work.ipynb`, and **Run All**.

Submissions 2–4 load the frozen split rather than re-splitting:

```python
manifest = pd.read_csv("../artifacts/split_manifest.csv")  # path relative to notebooks/
train_df = manifest[manifest.split == "train"]
val_df   = manifest[manifest.split == "val"]
test_df  = manifest[manifest.split == "test"]
RANDOM_SEED = 42
```

---

## Experimental protocol

Held fixed across all submissions so that comparisons are attributable:

| Setting | Value |
|---|---|
| Split | 70 / 15 / 15, grouped by duplicate identity, stratified by class |
| Random seed | 42 |
| Input size | 224 × 224 |
| Normalisation | ImageNet channel statistics |
| Augmentation | ±10° rotation, ±20% brightness/contrast, random resized crop (0.8–1.0) |
| **Horizontal flip** | **Not applied** — see below |

> **Why no horizontal flip.** It is the augmentation most people apply reflexively, and it is wrong here. A sweep and a leg glance are defined partly by which side of the body the ball is played to. Mirroring the image changes the correct label.

The Submission 4 comparison holds architecture, split, seed, optimizer, learning rate, batch size and epoch budget constant, varying **only** the presence of the attention module, so any measured difference is attributable to attention.

---

## Related work

Six papers surveyed in full in the Submission 1 notebook and report. Summary of the landscape:

- The field is **overwhelmingly video-based**. Of the surveyed work, only Dey & Biswas (2024) and Foysal et al. (2019) operate on single images, and the latter classifies bowlers rather than shots.
- **Reported accuracies are unreliable.** See [Known issue](#known-issue-duplicate-leakage) above.
- **Attention appears but is never isolated** — prior work introduces attention alongside other architectural changes, so its individual contribution cannot be separated. Submission 4 isolates it.

Key references:

1. Sen, Deb, Dhar & Koshiba (2021), *Sensors* — [CricShotClassify](https://pmc.ncbi.nlm.nih.gov/articles/PMC8072636)
2. Kang (2025) — [Modern Deep Learning Approaches for Cricket Shot Classification](https://arxiv.org/abs/2510.09187)
3. Dey & Biswas (2024), *Int. J. Engineering* — [Shot-ViT](https://www.ije.ir/article_195819.html)
4. Foysal, Islam, Karim & Neehal (2019) — [Shot-Net / cricket bowler identification](https://arxiv.org/abs/1909.01228)

---

## Licence

Code in this repository is MIT licensed. The dataset is **not** covered by that licence — see its Kaggle page for terms before redistributing any part of it.
