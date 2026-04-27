# Speech Emotion Recognition - Baseline Model
# Using 1D CNN on RAVDESS dataset

import os
import glob
import warnings
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Dense, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['figure.dpi'] = 100

print(f"NumPy     : {np.__version__}")
print(f"Pandas    : {pd.__version__}")
print(f"Librosa   : {librosa.__version__}")
print(f"TensorFlow: {tf.__version__}")
print("\nStarting SER pipeline...\n")


# Parsing RAVDESS filenames to get emotion labels 

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

DATASET_PATH = 'dataset'

def parse_ravdess_files(dataset_path):
    records = []
    wav_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)

    for path in wav_files:
        fname = os.path.basename(path)
        parts = fname.replace('.wav', '').split('-')
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        intensity = 'normal' if parts[3] == '01' else 'strong'
        actor = int(parts[6])
        gender = 'female' if actor % 2 == 0 else 'male'
        emotion = EMOTION_MAP.get(emotion_code, 'unknown')
        records.append({
            'file_path': path, 'emotion': emotion,
            'intensity': intensity, 'actor': actor, 'gender': gender
        })
    return pd.DataFrame(records)

df = parse_ravdess_files(DATASET_PATH)
print(f"Total files loaded : {len(df)}")
print(f"Unique emotions    : {df['emotion'].nunique()}")
print(f"Unique actors      : {df['actor'].nunique()}")
print(f"\nEmotion distribution:\n{df['emotion'].value_counts()}\n")


# Dataset visualization 

def plot_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RAVDESS Dataset Overview', fontsize=15, fontweight='bold')

    emotion_counts = df['emotion'].value_counts()
    sns.barplot(x=emotion_counts.values, y=emotion_counts.index,
                palette='viridis', ax=axes[0])
    axes[0].set_title('Emotion Class Distribution')
    axes[0].set_xlabel('Number of Samples')
    for i, v in enumerate(emotion_counts.values):
        axes[0].text(v + 1, i, str(v), va='center', fontsize=10)

    gender_counts = df['gender'].value_counts()
    axes[1].pie(gender_counts.values, labels=gender_counts.index,
                autopct='%1.1f%%', colors=['#5B9BD5', '#ED7D31'],
                startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Gender Distribution')

    intensity_counts = df['intensity'].value_counts()
    axes[2].pie(intensity_counts.values, labels=intensity_counts.index,
                autopct='%1.1f%%', colors=['#70AD47', '#FFC000'],
                startangle=90, textprops={'fontsize': 12})
    axes[2].set_title('Emotional Intensity Split')

    plt.tight_layout()
    plt.savefig('figures/plot_01_dataset_overview.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_01_dataset_overview.png")

plot_dataset_overview(df)


# Waveform and Mel spectrogram for each emotion.
# Using mel spectrogram to visually check that emotions look different in the audio.

def plot_waveforms_and_spectrograms(df, n_emotions=8):
    emotions = df['emotion'].unique()[:n_emotions]
    fig, axes = plt.subplots(n_emotions, 2, figsize=(16, n_emotions * 2.5))
    fig.suptitle('Waveform & Mel Spectrogram per Emotion', fontsize=14, fontweight='bold')

    for i, emotion in enumerate(emotions):
        sample_path = df[df['emotion'] == emotion]['file_path'].iloc[0]
        y, sr = librosa.load(sample_path, sr=22050, duration=3.0)

        # waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[i][0], color='steelblue', alpha=0.8)
        axes[i][0].set_title(f'{emotion.upper()} — Waveform', fontsize=10)
        axes[i][0].set_xlabel('Time (s)')
        axes[i][0].set_ylabel('Amplitude')

        # mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                       x_axis='time', y_axis='mel',
                                       ax=axes[i][1], cmap='magma')
        axes[i][1].set_title(f'{emotion.upper()} — Mel Spectrogram', fontsize=10)
        fig.colorbar(img, ax=axes[i][1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('figures/plot_02_waveforms_spectrograms.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_02_waveforms_spectrograms.png")

plot_waveforms_and_spectrograms(df)


# Feature extraction
# extracting 3 types of features and concatenating them:
# MFCCs (40) + Chroma (12) + Mel spectrogram (128) = 180 total

def extract_features(file_path, sr=22050, n_mfcc=40, n_mels=128):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=3.0)

        # pad/trim to exactly 3 seconds
        target_len = sr * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_mean = np.mean(mel, axis=1)

        return np.concatenate([mfccs_mean, chroma_mean, mel_mean])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


print("Extracting features from all 1440 files...")
features, labels = [], []
for idx, row in df.iterrows():
    feat = extract_features(row['file_path'])
    if feat is not None:
        features.append(feat)
        labels.append(row['emotion'])

X = np.array(features)
y_raw = np.array(labels)
print(f"Feature matrix shape: {X.shape}")
print(f"Feature vector size : {X.shape[1]} (40 MFCCs + 12 Chroma + 128 Mel)")


# MFCC heatmap to show that features are different per emotion.

def plot_mfcc_heatmap(X, y_raw, n_mfcc=40):
    feat_df = pd.DataFrame(X[:, :n_mfcc], columns=[f'MFCC_{i+1}' for i in range(n_mfcc)])
    feat_df['emotion'] = y_raw
    mfcc_means = feat_df.groupby('emotion').mean()

    plt.figure(figsize=(18, 6))
    sns.heatmap(mfcc_means, cmap='coolwarm', annot=False, linewidths=0.3)
    plt.title('Mean MFCC Values per Emotion Class', fontsize=13, fontweight='bold')
    plt.xlabel('MFCC Coefficient Index')
    plt.ylabel('Emotion')
    plt.tight_layout()
    plt.savefig('figures/plot_03_mfcc_heatmap.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_03_mfcc_heatmap.png")

plot_mfcc_heatmap(X, y_raw)


# Data preparation
# encode labels, scale features, split into train/test

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
y_cat = to_categorical(y_encoded)
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {list(le.classes_)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# reshaping for Conv1D: (samples, timesteps, 1)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")


#  Building the 1D CNN 

def build_1d_cnn(input_shape, n_classes):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', padding='same',
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        GlobalAveragePooling1D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_1d_cnn(input_shape=X_train.shape[1:], n_classes=n_classes)
model.summary()


# Training the model

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, batch_size=32,
    callbacks=callbacks, verbose=1
)
print("Training complete.")




def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Categorical Crossentropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('figures/plot_04_training_history.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_04_training_history.png")

plot_training_history(history)


# Evaluating on test set.

def evaluate_model(model, X_test, y_test, le):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix — Test Set', fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
    axes[1].set_title('Normalised')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('figures/plot_05_confusion_matrix.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_05_confusion_matrix.png")
    return y_pred, y_true

y_pred, y_true = evaluate_model(model, X_test, y_test, le)


#Per-class accuracy chart 

def plot_per_class_accuracy(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    acc_df = pd.DataFrame({
        'Emotion': le.classes_, 'Accuracy': per_class_acc
    }).sort_values('Accuracy', ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(acc_df['Emotion'], acc_df['Accuracy'],
                    color=sns.color_palette('viridis', len(acc_df)))
    plt.axvline(x=acc_df['Accuracy'].mean(), color='red',
                linestyle='--', linewidth=1.5, label=f"Mean = {acc_df['Accuracy'].mean():.2f}")
    plt.xlabel('Accuracy')
    plt.title('Per-Class Recognition Accuracy', fontsize=13, fontweight='bold')
    plt.xlim(0, 1.05)
    for bar, val in zip(bars, acc_df['Accuracy']):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/plot_06_per_class_accuracy.png', bbox_inches='tight')
    plt.show()
    print("Saved: figures/plot_06_per_class_accuracy.png")

plot_per_class_accuracy(y_true, y_pred, le)


# Save model 

model.save('ser_1d_cnn_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Model, scaler, and label encoder saved.")



def predict_emotion(file_path, model, scaler, le):
    feat = extract_features(file_path)
    if feat is None:
        return None

    feat_scaled = scaler.transform(feat.reshape(1, -1))
    feat_cnn = feat_scaled.reshape(1, feat_scaled.shape[1], 1)

    probs = model.predict(feat_cnn, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.classes_[pred_idx]

    print(f"\nFile     : {os.path.basename(file_path)}")
    print(f"Predicted: {pred_label.upper()}  (confidence: {probs[pred_idx]*100:.1f}%)")
    for emotion, prob in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"  {emotion:<12} {prob*100:5.1f}%  {bar}")

    plt.figure(figsize=(10, 4))
    colors = ['#2ecc71' if e == pred_label else '#95a5a6' for e in le.classes_]
    plt.bar(le.classes_, probs, color=colors, edgecolor='white')
    plt.title(f'Prediction: "{pred_label.upper()}" ({probs[pred_idx]*100:.1f}%)',
              fontsize=13, fontweight='bold')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('figures/plot_07_prediction_demo.png', bbox_inches='tight')
    plt.show()
    return pred_label

# testing on a few random samples.
sample_files = df['file_path'].sample(3, random_state=7).tolist()
for f in sample_files:
    true_label = df[df['file_path'] == f]['emotion'].values[0]
    print(f"\nTrue label: {true_label.upper()}")
    predict_emotion(f, model, scaler, le)

print("\n" + "="*60)
print("BASELINE PIPELINE COMPLETE")
print("="*60)
