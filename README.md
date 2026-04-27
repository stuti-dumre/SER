# Speech Emotion Recognition using 1D CNN

Classifies 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from audio using a 1D Convolutional Neural Network trained on the RAVDESS dataset.

**Best result:** 90.71% accuracy, 90.81% F1 score

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download RAVDESS from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Place the Actor folders inside a `dataset/` directory:
```
dataset/
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

## How to Run

**Baseline pipeline** (generates visualizations + trains baseline model):
```bash
python main.py
```

**Experiments** (runs 5 experiments with augmentation, saves best model):
```bash
python experiments.py
```

## Project Structure

```
├── main.py                  # Baseline pipeline (feature extraction, CNN, evaluation)
├── experiments.py           # 5 experiments comparing approaches (augmentation, temporal features)
├── dataset/                 # RAVDESS audio files (24 actor folders)
├── figures/                 # Generated plots
├── best_ser_model.keras     # Best trained model (Exp4, 90.71%)
├── best_scaler.pkl          # StandardScaler for feature normalization
├── best_label_encoder.pkl   # Label encoder (emotion <-> integer mapping)
├── experiment_results.json  # Metrics from all experiments
├── requirements.txt         # Python dependencies
└── README.md
```

## Experiments Summary

| # | Approach | Accuracy | F1 Score |
|---|----------|----------|----------|
| 1 | Baseline (mean features, no augmentation) | 42.71% | 41.93% |
| 2 | Enhanced features (deltas, ZCR, RMS) | 42.01% | 40.21% |
| 3 | Enhanced features + data augmentation (4x) | 67.01% | 67.12% |
| 4 | **Temporal MFCCs + deep CNN + augmentation** | **90.71%** | **90.81%** |
| 5 | Temporal + 4 emotion classes | 89.06% | 89.34% |

## Key Findings

- Data augmentation (noise, pitch shift, time stretch) improved accuracy from 42% to 67%
- Keeping temporal MFCC sequences instead of averaging improved from 67% to 91%
- Low-arousal emotions (neutral, calm, sad) are the hardest to distinguish
