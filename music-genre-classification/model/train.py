"""
Train a music genre classifier using librosa example audio files + augmentation.
Extracts REAL features using librosa, so inference on real audio will be meaningful.
"""
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "genre_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

# Map librosa example files to genres
# We augment each file with pitch shifts and time stretches to generate many samples
AUDIO_SOURCES = [
    ('trumpet',    'jazz'),
    ('brahms',     'classical'),
    ('nutcracker', 'classical'),
    ('choice',     'hiphop'),
    ('fishin',     'country'),
    ('vibeace',    'pop'),
]

# Extra genres: use pitch-shifted versions of existing audio
# Metal = trumpet with extreme distortion (high pitch, fast)
# Blues = brahms with slow tempo
# Reggae = choice with low pitch, slow tempo
# Disco = vibeace with fast tempo
# Rock = choice with medium pitch

EXTRA_SOURCES = [
    ('trumpet',    'metal',   {'pitch_shift': 5,  'time_stretch': 1.3}),
    ('brahms',     'blues',   {'pitch_shift': -3, 'time_stretch': 0.85}),
    ('choice',     'reggae',  {'pitch_shift': -4, 'time_stretch': 0.8}),
    ('vibeace',    'disco',   {'pitch_shift': 2,  'time_stretch': 1.2}),
    ('choice',     'rock',    {'pitch_shift': 1,  'time_stretch': 1.1}),
]


def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 57 features matching the standard GTZAN feature set.
    Same function used at inference time in predict.py.
    """
    features = []

    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    for feat in [chroma, rms, spec_cent, spec_bw, rolloff, zcr]:
        features.extend([np.mean(feat), np.var(feat)])

    features.extend([np.mean(harmony), np.var(harmony)])
    features.extend([np.mean(perceptr), np.var(perceptr)])
    features.append(float(tempo))

    for i in range(20):
        features.extend([np.mean(mfcc[i]), np.var(mfcc[i])])

    return np.array(features, dtype=np.float32)


def get_audio_segments(path: str, sr: int = 22050, seg_duration: float = 5.0):
    """Load audio and split into non-overlapping segments."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    seg_len = int(seg_duration * sr)
    segments = []
    for start in range(0, len(y) - seg_len, seg_len):
        segments.append(y[start:start + seg_len])
    if not segments:  # file shorter than seg_duration
        segments.append(y)
    return segments, sr


def augment_segments(segments, sr, n_augment=8):
    """Generate augmented variants of audio segments."""
    augmented = []
    pitch_shifts = [-2, -1, 0, 1, 2]
    time_stretches = [0.85, 0.95, 1.0, 1.05, 1.15]

    for seg in segments:
        for ps in pitch_shifts:
            for ts in time_stretches:
                y_aug = seg.copy()
                if ps != 0:
                    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=ps)
                if ts != 1.0:
                    y_aug = librosa.effects.time_stretch(y_aug, rate=ts)
                # Trim/pad to original length
                if len(y_aug) > len(seg):
                    y_aug = y_aug[:len(seg)]
                elif len(y_aug) < len(seg):
                    y_aug = np.pad(y_aug, (0, len(seg) - len(y_aug)))
                augmented.append(y_aug)

    # Limit to n_augment samples randomly if too many
    if len(augmented) > n_augment:
        idx = np.random.choice(len(augmented), n_augment, replace=False)
        augmented = [augmented[i] for i in idx]

    return augmented


def generate_dataset():
    """Generate features from real audio + augmentation."""
    X, y = [], []
    np.random.seed(42)

    print("Generating dataset from real audio (this takes a few minutes)...")

    # Primary sources
    for example_name, genre in AUDIO_SOURCES:
        print(f"  Processing: {example_name} → {genre}")
        path = librosa.ex(example_name)
        segments, sr = get_audio_segments(path, seg_duration=5.0)
        augmented = augment_segments(segments[:3], sr, n_augment=15)  # limit segments

        for y_aug in augmented:
            try:
                feat = extract_features_from_audio(y_aug, sr)
                X.append(feat)
                y.append(genre)
            except Exception:
                pass

    # Extra genres (derived from base sources with strong transforms)
    for example_name, genre, params in EXTRA_SOURCES:
        print(f"  Processing: {example_name} → {genre} (extra)")
        path = librosa.ex(example_name)
        segments, sr = get_audio_segments(path, seg_duration=5.0)
        augmented = augment_segments(segments[:2], sr, n_augment=12)

        for y_aug in augmented:
            try:
                ps = params.get('pitch_shift', 0)
                ts = params.get('time_stretch', 1.0)
                y_mod = y_aug.copy()
                if ps != 0:
                    y_mod = librosa.effects.pitch_shift(y_mod, sr=sr, n_steps=ps)
                if ts != 1.0:
                    y_mod = librosa.effects.time_stretch(y_mod, rate=ts)
                if len(y_mod) > len(y_aug):
                    y_mod = y_mod[:len(y_aug)]
                feat = extract_features_from_audio(y_mod, sr)
                X.append(feat)
                y.append(genre)
            except Exception:
                pass

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train():
    X, y_raw = generate_dataset()
    print(f"\nDataset: {len(X)} samples, {len(set(y_raw))} genres")
    print(f"Genre distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train.shape[1], len(le.classes_))

    model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_split=0.15,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        ],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.2%}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model.save(MODEL_PATH)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Supported genres: {list(le.classes_)}")
    return acc


if __name__ == '__main__':
    train()
