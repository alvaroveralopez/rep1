import pandas as pd
import librosa
import numpy as np
import os
from tabulate import tabulate


# ------------------------------------------------------------------------------------
#                                SPECTRAL FEATURES
# ------------------------------------------------------------------------------------

# chroma_stft ------------------------------------------------------------------------

def extract_chroma_stft(y, sr):
    try:
        chroma_stft= librosa.feature.chroma_stft(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return chroma_stft
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_cqt ------------------------------------------------------------------------

def extract_chroma_cqt(y, sr):
    try:
        chroma_cqt= librosa.feature.chroma_cqt(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return chroma_cqt
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_cens ------------------------------------------------------------------------

def extract_chroma_cens(y, sr):
    try:
        chroma_cens= librosa.feature.chroma_cens(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return chroma_cens
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_vqt ------------------------------------------------------------------------

def extract_chroma_vqt(y, sr):
    try:
        chroma_vqt= librosa.feature.chroma_vqt(y=y, sr=sr, intervals='equal')
        # print(f"Procesado: {audio_path}")
        return chroma_vqt
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]


# melspectrogram ------------------------------------------------------------------------
def extract_melspectrogram(y, sr):
    try:
        melspectrogram= librosa.feature.melspectrogram(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return melspectrogram
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# MFCC ------------------------------------------------------------------------
def extract_mfcc(y, sr):
    try:
        mfcc= librosa.feature.mfcc(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return mfcc
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# MFCC ------------------------------------------------------------------------
def extract_rms(y, sr):
    try:
        rms= librosa.feature.rms(y=y)
        # print(f"Procesado: {audio_path}")
        return rms
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_centroid --------------------------------------------------------------

def extract_spectral_centroid(y, sr):
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_centroid
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_bandwidth ------------------------------------------------------------
def extract_spectral_bandwidth(y, sr):
    try:
        spectral_bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_bandwidth
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_contrast --------------------------------------------------------------

def extract_spectral_contrast(y, sr):
    try:
        # S = np.abs(librosa.stft(y))
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_contrast
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]


# spectral_flatness ---------------------------------------------------------------------

def extract_spectral_flatness(y, sr):
    try:
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        # print(f"Procesado: {audio_path}")
        return spectral_flatness
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_rollof ---------------------------------------------------------------------

def extract_spectral_rolloff(y, sr):
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_rolloff
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# poly_features ---------------------------------------------------------------------

def extract_poly_features(y, sr):
    try:
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return poly_features
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# tonnetz ---------------------------------------------------------------------

def extract_tonnetz(y, sr):
    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return tonnetz
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# zero_crossing_rate ---------------------------------------------------------------------
def extract_zero_crossing_rate(y, sr):
    try:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        # print(f"Procesado: {audio_path}")
        return zero_crossing_rate
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return [None]

# ALL SPECTRAL FEATURES-----------------------------------------------------------

def extract_spectral_features(audio_path):

    y, sr = librosa.load(audio_path)
    chroma_stft = extract_chroma_stft(y, sr)
    chroma_cqt = extract_chroma_cqt(y, sr)
    chroma_cens = extract_chroma_cens(y, sr)
    chroma_vqt = extract_chroma_vqt(y, sr)
    melspectrogram = extract_melspectrogram(y, sr)
    mfcc = extract_mfcc(y, sr)
    rms = extract_rms(y, sr)
    spectral_centroid = extract_spectral_centroid(y, sr)
    spectral_bandwidth = extract_spectral_bandwidth(y, sr)
    spectral_contrast = extract_spectral_contrast(y, sr)
    spectral_flatness = extract_spectral_flatness(y, sr)
    spectral_rolloff = extract_spectral_rolloff(y, sr)
    poly_features = extract_poly_features(y, sr)
    tonnetz = extract_tonnetz(y, sr)
    zero_crossing_rate = extract_zero_crossing_rate(y, sr)

    return (chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc,
     rms, spectral_centroid, spectral_bandwidth, spectral_contrast,
     spectral_flatness, spectral_rolloff, poly_features, tonnetz,
     zero_crossing_rate)



# ----------------------------------------------------------------------------------------

# CSV de los audios
input_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv"
output_folder = "C:/Users/alvar/PycharmProjects/Proyecto2/features_npy"

# Leer el archivo CSV original
data = pd.read_csv(input_csv_path)

all_data = []

# Recorrer las filas del DataFrame
for index, row in data.iterrows():
    audio_path = row["path"]
    id_audio = row["id_audio"]

    print(f"Procesando...")

    (chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc,
     rms, spectral_centroid, spectral_bandwidth, spectral_contrast,
     spectral_flatness, spectral_rolloff, poly_features, tonnetz,
     zero_crossing_rate) = extract_spectral_features(audio_path)

    print(f"Procesado: {audio_path}")

    chroma_stft_file = f"{output_folder}/chroma_stft/{id_audio}.npy"
    os.makedirs(os.path.dirname(chroma_stft_file), exist_ok=True)
    np.save(chroma_stft_file, chroma_stft)

    chroma_cqt_file = f"{output_folder}/chroma_cqt/{id_audio}.npy"
    os.makedirs(os.path.dirname(chroma_cqt_file), exist_ok=True)
    np.save(chroma_cqt_file, chroma_cqt)

    chroma_cens_file = f"{output_folder}/chroma_cens/{id_audio}.npy"
    os.makedirs(os.path.dirname(chroma_cens_file), exist_ok=True)
    np.save(chroma_cens_file, chroma_cens)

    chroma_vqt_file = f"{output_folder}/chroma_vqt/{id_audio}.npy"
    os.makedirs(os.path.dirname(chroma_vqt_file), exist_ok=True)
    np.save(chroma_vqt_file, chroma_vqt)

    melspectrogram_file = f"{output_folder}/melspectrogram/{id_audio}.npy"
    os.makedirs(os.path.dirname(melspectrogram_file), exist_ok=True)
    np.save(melspectrogram_file, melspectrogram)

    mfcc_file = f"{output_folder}/mfcc/{id_audio}.npy"
    os.makedirs(os.path.dirname(mfcc_file), exist_ok=True)
    np.save(mfcc_file, mfcc)

    rms_file = f"{output_folder}/rms/{id_audio}.npy"
    os.makedirs(os.path.dirname(rms_file), exist_ok=True)
    np.save(rms_file, rms)

    spectral_centroid_file = f"{output_folder}/spectral_centroid/{id_audio}.npy"
    os.makedirs(os.path.dirname(spectral_centroid_file), exist_ok=True)
    np.save(spectral_centroid_file, spectral_centroid)

    spectral_bandwidth_file = f"{output_folder}/spectral_bandwidth/{id_audio}.npy"
    os.makedirs(os.path.dirname(spectral_bandwidth_file), exist_ok=True)
    np.save(spectral_bandwidth_file, spectral_bandwidth)

    spectral_contrast_file = f"{output_folder}/spectral_contrast/{id_audio}.npy"
    os.makedirs(os.path.dirname(spectral_contrast_file), exist_ok=True)
    np.save(spectral_contrast_file, spectral_contrast)

    spectral_flatness_file = f"{output_folder}/spectral_flatness/{id_audio}.npy"
    os.makedirs(os.path.dirname(spectral_flatness_file), exist_ok=True)
    np.save(spectral_flatness_file, spectral_flatness)

    spectral_rolloff_file = f"{output_folder}/spectral_rolloff/{id_audio}.npy"
    os.makedirs(os.path.dirname(spectral_rolloff_file), exist_ok=True)
    np.save(spectral_rolloff_file, spectral_rolloff)

    poly_features_file = f"{output_folder}/poly_features/{id_audio}.npy"
    os.makedirs(os.path.dirname(poly_features_file), exist_ok=True)
    np.save(poly_features_file, poly_features)

    tonnetz_file = f"{output_folder}/tonnetz/{id_audio}.npy"
    os.makedirs(os.path.dirname(tonnetz_file), exist_ok=True)
    np.save(tonnetz_file, tonnetz)

    zero_crossing_rate_file = f"{output_folder}/zero_crossing_rate/{id_audio}.npy"
    os.makedirs(os.path.dirname(zero_crossing_rate_file), exist_ok=True)
    np.save(zero_crossing_rate_file, zero_crossing_rate)

    new_row = {
        "id_audio": id_audio,
        "chroma_stft": chroma_stft_file,
        "chroma_cqt": chroma_cqt_file,
        "chroma_cens": chroma_cens_file,
        "chroma_vqt": chroma_vqt_file,
        "melspectrogram": melspectrogram_file,
        "mfcc": mfcc_file,
        "rms": rms_file,
        "spectral_centroid": spectral_centroid_file,
        "spectral_bandwidth": spectral_bandwidth_file,
        "spectral_contrast": spectral_contrast_file,
        "spectral_flatness": spectral_flatness_file,
        "spectral_rolloff": spectral_rolloff_file,
        "poly_features": poly_features_file,
        "tonnetz": tonnetz_file,
        "zero_crossing_rate": zero_crossing_rate_file

    }
    all_data.append(new_row)


feature_df = pd.DataFrame(all_data)
output_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/spectral_features.csv"
feature_df.to_csv(output_csv_path, index=False)

print(feature_df.head)