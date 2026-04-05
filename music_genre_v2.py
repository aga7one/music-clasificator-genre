# -*- coding: utf-8 -*-
"""Clasificador de Género Musical - Versión Mejorada

Mejoras implementadas:
- Más features: MFCCs (40), chroma, spectral contrast, tonnetz, ZCR
- Bidirectional LSTM apilado con BatchNormalization
- Data augmentation (pitch shift, ruido, time stretch)
- EarlyStopping + ReduceLROnPlateau
- División estratificada del dataset
"""

# !pip install librosa

import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

# !wget https://web.archive.org/web/20220328223413/ftp://opihi.cs.uvic.ca/sound/genres.tar.gz
# !tar -xzf genres.tar.gz


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # memoria dinámica para no reservar toda la VRAM de golpe
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    print(f"GPU detectada: {gpus[0].name}")
else:
    print("No se detectó GPU, usando CPU")

DATASET_PATH = "genres"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40
MAX_LEN = 1300
VALID_EXTENSIONS = (".wav", ".au", ".mp3")

print("--->", DATASET_PATH)
print("--->", os.listdir(DATASET_PATH))


# ─── Extracción de features ───────────────────────────────────────────────────

def extract_features_from_signal(signal, sr, n_mfcc=N_MFCC):
    """Extrae MFCCs, chroma, spectral contrast, tonnetz y ZCR de una señal."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)                        # (n_mfcc, T)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)                               # (12, T)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)              # (7, T)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)        # (6, T)
    zcr = librosa.feature.zero_crossing_rate(signal)                                    # (1, T)

    features = np.vstack([mfcc, chroma, spectral_contrast, tonnetz, zcr])  # (66, T)
    return features.T  # (T, 66)


def normalize_signal(signal):
    """Fuerza la señal a SAMPLES_PER_TRACK frames."""
    if len(signal) >= SAMPLES_PER_TRACK:
        return signal[:SAMPLES_PER_TRACK]
    return np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))


# ─── Data augmentation ────────────────────────────────────────────────────────

def augment_signal(signal, sr):
    """Aplica transformaciones aleatorias para aumentar el dataset."""
    choice = np.random.randint(0, 3)
    if choice == 0:
        # Pitch shift ±2 semitonos
        steps = np.random.uniform(-2, 2)
        signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)
    elif choice == 1:
        # Ruido gaussiano leve
        signal = signal + 0.005 * np.random.randn(len(signal))
    else:
        # Time stretch entre 0.9x y 1.1x
        rate = np.random.uniform(0.9, 1.1)
        signal = librosa.effects.time_stretch(signal, rate=rate)
    return normalize_signal(signal)


# ─── Carga del dataset ────────────────────────────────────────────────────────

def load_dataset(dataset_path, augment=True):
    X, y = [], []

    for genre in sorted(os.listdir(dataset_path)):
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"Procesando: {genre}")

        for file in os.listdir(genre_path):
            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            file_path = os.path.join(genre_path, file)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = normalize_signal(signal)

                # muestra original
                X.append(extract_features_from_signal(signal, sr))
                y.append(genre)

                # muestra aumentada
                if augment:
                    aug_signal = augment_signal(signal, sr)
                    X.append(extract_features_from_signal(aug_signal, sr))
                    y.append(genre)

            except Exception as e:
                print(f"  Error en {file_path}: {e}")

    return np.array(X), np.array(y)


# ─── Pipeline principal ───────────────────────────────────────────────────────

X, y = load_dataset(DATASET_PATH, augment=True)
print(f"Total muestras (con augmentation): {len(X)}")
print(f"Shape de una muestra: {X[0].shape}")

# Padding/truncating a MAX_LEN frames
X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post', dtype='float32')

# Encoding de etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
n_classes = y_categorical.shape[1]
n_features = X_padded.shape[2]

print(f"Clases: {le.classes_}")
print(f"Shape final X: {X_padded.shape}")

# División estratificada para balancear clases
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")


# ─── Modelo: Bidirectional LSTM apilado ──────────────────────────────────────

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(MAX_LEN, n_features)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ─── Callbacks ────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)
]


# ─── Entrenamiento ────────────────────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=callbacks
)


# ─── Evaluación ───────────────────────────────────────────────────────────────

loss, acc = model.evaluate(X_test, y_test)
print(f"\nAccuracy en test: {acc:.4f} ({acc*100:.2f}%)")

# Curvas de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('Loss')
ax2.legend()
plt.tight_layout()
plt.show()


# ─── Predicción ───────────────────────────────────────────────────────────────

def predict_genre(file_path, model, label_encoder):
    """Predice el género de un archivo de audio."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    signal = normalize_signal(signal)
    features = extract_features_from_signal(signal, sr)
    features_padded = pad_sequences([features], maxlen=MAX_LEN, padding='post',
                                     truncating='post', dtype='float32')
    prediction = model.predict(features_padded, verbose=0)
    genre = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)
    print(f"Género predicho: {genre} (confianza: {confidence:.2%})")
    return genre

# Ejemplo de uso:
predict_genre("test.wav", model, le)

model.save("music_genre_classifier.h5")
print("Modelo guardado en music_genre_classifier.h5")
