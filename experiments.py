# SER Tuning and running experiments to find the best model.
# Also adding data augmentation 

import os, glob, warnings, time, json, pickle
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Dense, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 100

print(f"TensorFlow: {tf.__version__}")
print("="*60)


#Load dataset 

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_dataset(path='dataset'):
    records = []
    for fp in glob.glob(os.path.join(path, '**', '*.wav'), recursive=True):
        parts = os.path.basename(fp).replace('.wav','').split('-')
        if len(parts) != 7: continue
        records.append({
            'file_path': fp,
            'emotion': EMOTION_MAP.get(parts[2], 'unknown'),
            'actor': int(parts[6])
        })
    return pd.DataFrame(records)

df = load_dataset()
print(f"Dataset: {len(df)} files, {df['emotion'].nunique()} emotions\n")


# Augmentation functions
# we have only 1440 samples so we created more training data

def augment_noise(y, noise_factor=0.005):
    """add small random noise to simulate background noise"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_pitch(y, sr, n_steps=2):
    """shift pitch up by 2 semitones to simulate different speakers"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_stretch(y, rate=1.1):
    """speed up by 10% to simulate faster speaking"""
    return librosa.effects.time_stretch(y, rate=rate)


# Feature extractors 

def extract_enhanced(path, sr=22050):
    """mean+std of MFCC, delta, delta2, chroma, mel, zcr, rms"""
    y, sr = librosa.load(path, sr=sr, duration=3.0)
    y = librosa.util.fix_length(y, size=sr*3)
    return _extract_from_signal(y, sr)

def _extract_from_signal(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = []
    for f in [mfcc, delta1, delta2, chroma, mel]:
        feats.append(np.mean(f, axis=1))
        feats.append(np.std(f, axis=1))
    feats.append(np.mean(zcr)); feats.append(np.std(zcr))
    feats.append(np.mean(rms)); feats.append(np.std(rms))
    return np.concatenate([np.atleast_1d(f) for f in feats])


def extract_temporal(path, sr=22050, n_mfcc=40, max_len=130):
    """keep full MFCC time-series instead of averaging"""
    y, sr = librosa.load(path, sr=sr, duration=3.0)
    y = librosa.util.fix_length(y, size=sr*3)
    return _temporal_from_signal(y, sr, n_mfcc, max_len)

def _temporal_from_signal(y, sr, n_mfcc=40, max_len=130):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (T, 40)
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:max_len]
    return mfcc


# Build augmented datasets 

def build_augmented_dataset_enhanced(df, sr=22050):
    print("  Building augmented dataset (enhanced features)...", flush=True)
    t0 = time.time()
    feats, labs = [], []
    for _, row in df.iterrows():
        y, sr_ = librosa.load(row['file_path'], sr=sr, duration=3.0)
        y = librosa.util.fix_length(y, size=sr*3)
        emotion = row['emotion']

        feats.append(_extract_from_signal(y, sr)); labs.append(emotion)
        feats.append(_extract_from_signal(augment_noise(y), sr)); labs.append(emotion)
        feats.append(_extract_from_signal(augment_pitch(y, sr, 2), sr)); labs.append(emotion)
        y_stretched = librosa.util.fix_length(augment_stretch(y, 1.1), size=sr*3)
        feats.append(_extract_from_signal(y_stretched, sr)); labs.append(emotion)

    print(f"  done ({time.time()-t0:.1f}s, {len(feats)} samples)")
    return np.array(feats), np.array(labs)


def build_augmented_dataset_temporal(df, sr=22050):
    print("  Building augmented dataset (temporal features)...", flush=True)
    t0 = time.time()
    feats, labs = [], []
    for _, row in df.iterrows():
        y, sr_ = librosa.load(row['file_path'], sr=sr, duration=3.0)
        y = librosa.util.fix_length(y, size=sr*3)
        emotion = row['emotion']

        feats.append(_temporal_from_signal(y, sr)); labs.append(emotion)
        feats.append(_temporal_from_signal(augment_noise(y), sr)); labs.append(emotion)
        feats.append(_temporal_from_signal(augment_pitch(y, sr, 2), sr)); labs.append(emotion)
        y_s = librosa.util.fix_length(augment_stretch(y, 1.1), size=sr*3)
        feats.append(_temporal_from_signal(y_s, sr)); labs.append(emotion)

    print(f"  done ({time.time()-t0:.1f}s, {len(feats)} samples)")
    return np.array(feats), np.array(labs)


# Model architectures 

def build_tuned_cnn(input_shape, n_classes):
    """wider CNN with lower learning rate"""
    m = Sequential([
        Conv1D(128, 5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(256, 5, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_temporal_cnn(input_shape, n_classes):
    """4-block CNN for temporal MFCC input (130x40)"""
    m = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_baseline_cnn(input_shape, n_classes):
    """same architecture as main.py for comparison"""
    m = Sequential([
        Conv1D(64, 5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m


 

def run_experiment(name, X, y_raw, model_builder, epochs=150, batch_size=32,
                   patience=25, is_temporal=False):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Samples: {len(y_raw)}, Features: {X.shape[1:]}")
    print(f"{'='*60}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    y_cat = to_categorical(y_enc)
    n_classes = len(le.classes_)

    # standardize features
    if not is_temporal:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_in = X_s.reshape(X_s.shape[0], X_s.shape[1], 1)
    else:
        # for temporal data, normalizing each MFCC column independently.
        orig = X.shape
        scaler = StandardScaler()
        X_in = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(orig)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_in, y_cat, test_size=0.2, random_state=42, stratify=y_enc)

    model = model_builder(X_tr.shape[1:], n_classes)

    cbs = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0)
    ]

    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                     epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=0)
    train_time = time.time() - t0

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    _, test_acc = model.evaluate(X_te, y_te, verbose=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy : {test_acc*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"  Epochs   : {len(hist.history['loss'])}")
    print(f"  Time     : {train_time:.1f}s")

    return {
        'name': name, 'accuracy': round(test_acc*100, 2),
        'f1_score': round(f1*100, 2), 'epochs': len(hist.history['loss']),
        'train_time': round(train_time, 1), 'params': model.count_params(),
        'history': hist.history, 'cm': cm, 'report': report,
        'classes': list(le.classes_), 'model': model, 'scaler': scaler, 'le': le
    }


#Run all experiments

results = []

# Exp 1: baseline - mean features, no augmentation
print("\n--- Extracting baseline features ---")
X1, y1 = [], []
for _, row in df.iterrows():
    y, sr = librosa.load(row['file_path'], sr=22050, duration=3.0)
    y = librosa.util.fix_length(y, size=22050*3)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1)
    X1.append(np.concatenate([mfcc, chroma, mel]))
    y1.append(row['emotion'])
X1, y1 = np.array(X1), np.array(y1)
print(f"  Baseline features: {X1.shape}")
r1 = run_experiment("Exp1: Baseline (mean features, original CNN, no augmentation)",
                    X1, y1, build_baseline_cnn, epochs=100, patience=15)
results.append(r1)

# Exp 2: enhanced features, tuned CNN, no augmentation
print("\n--- Extracting enhanced features ---")
X2, y2 = [], []
for _, row in df.iterrows():
    X2.append(extract_enhanced(row['file_path']))
    y2.append(row['emotion'])
X2, y2 = np.array(X2), np.array(y2)
print(f"  Enhanced features: {X2.shape}")
r2 = run_experiment("Exp2: Enhanced features + tuned CNN (no augmentation)",
                    X2, y2, build_tuned_cnn, epochs=150, patience=25)
results.append(r2)

# Exp 3: enhanced features + augmentation (4x data)
X3, y3 = build_augmented_dataset_enhanced(df)
r3 = run_experiment("Exp3: Enhanced features + tuned CNN + augmentation (4x data)",
                    X3, y3, build_tuned_cnn, epochs=150, patience=25)
results.append(r3)

# Exp 4: temporal MFCCs + deep CNN + augmentation
X4, y4 = build_augmented_dataset_temporal(df)
r4 = run_experiment("Exp4: Temporal MFCCs + deep CNN + augmentation (4x data)",
                    X4, y4, build_temporal_cnn, epochs=150, patience=25, is_temporal=True)
results.append(r4)

# Exp 5: same as exp4 but with only 4 emotion classes

print("\n--- Building 4-class dataset ---")
MERGE_MAP = {
    'neutral': 'calm', 'calm': 'calm',
    'happy': 'happy', 'surprised': 'happy',
    'sad': 'sad',
    'angry': 'angry', 'fearful': 'angry', 'disgust': 'angry'
}
df5 = df.copy()
df5['emotion'] = df5['emotion'].map(MERGE_MAP)
X5, y5 = build_augmented_dataset_temporal(df5)
r5 = run_experiment("Exp5: Temporal + augmentation + 4 classes (calm/happy/sad/angry)",
                    X5, y5, build_temporal_cnn, epochs=150, patience=25, is_temporal=True)
results.append(r5)


# Generating comparison plots 

print("\n" + "="*60)
print("GENERATING COMPARISON PLOTS")
print("="*60)

# accuracy + f1 bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('All Experiments — Accuracy & F1 Comparison', fontsize=14, fontweight='bold')
names_short = [f'Exp{i+1}' for i in range(len(results))]
accs = [r['accuracy'] for r in results]
f1s = [r['f1_score'] for r in results]
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

bars1 = axes[0].bar(names_short, accs, color=colors[:len(results)], edgecolor='white')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Test Accuracy')
axes[0].set_ylim(0, 100)
for b, v in zip(bars1, accs):
    axes[0].text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold')

bars2 = axes[1].bar(names_short, f1s, color=colors[:len(results)], edgecolor='white')
axes[1].set_ylabel('Weighted F1 (%)')
axes[1].set_title('Weighted F1 Score')
axes[1].set_ylim(0, 100)
for b, v in zip(bars2, f1s):
    axes[1].text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/tuned_01_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: figures/tuned_01_comparison.png")

# training curves for all experiments
n = len(results)
cols = min(3, n)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
fig.suptitle('Training History — All Experiments', fontsize=14, fontweight='bold')
axes_flat = axes.flatten() if n > 1 else [axes]
for i, r in enumerate(results):
    ax = axes_flat[i]
    ax.plot(r['history']['accuracy'], label='Train', linewidth=1.5)
    ax.plot(r['history']['val_accuracy'], label='Val', linewidth=1.5, linestyle='--')
    ax.set_title(f"Exp{i+1}: {r['accuracy']}%", fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
plt.savefig('figures/tuned_02_training_curves.png', bbox_inches='tight')
plt.close()
print("Saved: figures/tuned_02_training_curves.png")

# confusion matrix for the best experiment
best = max(results, key=lambda r: r['accuracy'])
fig, ax = plt.subplots(figsize=(9, 7))
cm_n = best['cm'].astype('float') / best['cm'].sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=best['classes'], yticklabels=best['classes'], ax=ax)
ax.set_title(f"Best: {best['name']}\nAccuracy: {best['accuracy']}%", fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('figures/tuned_03_best_confusion.png', bbox_inches='tight')
plt.close()
print("Saved: figures/tuned_03_best_confusion.png")

# save best model and artifacts
best['model'].save('best_ser_model.keras')
with open('best_scaler.pkl', 'wb') as f: pickle.dump(best['scaler'], f)
with open('best_label_encoder.pkl', 'wb') as f: pickle.dump(best['le'], f)

# save results to json
summary = []
for r in results:
    summary.append({
        'name': r['name'], 'accuracy': r['accuracy'], 'f1_score': r['f1_score'],
        'epochs': r['epochs'], 'train_time_sec': r['train_time'], 'params': r['params'],
        'per_class': {c: round(r['report'][c]['f1-score']*100, 1) for c in r['classes']}
    })
with open('experiment_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
for r in results:
    print(f"  {r['name'][:55]:<55} Acc={r['accuracy']:5.2f}%  F1={r['f1_score']:5.2f}%")
print(f"\n  >>> BEST: {best['name']} — {best['accuracy']}% <<<")
