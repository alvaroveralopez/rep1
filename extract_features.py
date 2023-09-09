import pandas as pd
import librosa
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
        print(f"chroma_stft Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_cqt ------------------------------------------------------------------------

def extract_chroma_cqt(y, sr):
    try:
        chroma_cqt= librosa.feature.chroma_cqt(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return chroma_cqt
    except Exception as e:
        print(f"chroma_cqt Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_cens ------------------------------------------------------------------------

def extract_chroma_cens(y, sr):
    try:
        chroma_cens= librosa.feature.chroma_cens(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return chroma_cens
    except Exception as e:
        print(f"chroma_cens Error al procesar el archivo {audio_path}: {e}")
        return [None]

# chroma_vqt ------------------------------------------------------------------------

def extract_chroma_vqt(y, sr):
    try:
        chroma_vqt= librosa.feature.chroma_vqt(y=y, sr=sr, intervals='equal')
        # print(f"Procesado: {audio_path}")
        return chroma_vqt
    except Exception as e:
        print(f"chroma_vqt Error al procesar el archivo {audio_path}: {e}")
        return [None]


# melspectrogram ------------------------------------------------------------------------
def extract_melspectrogram(y, sr):
    try:
        melspectrogram= librosa.feature.melspectrogram(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return melspectrogram
    except Exception as e:
        print(f"melspectrogram Error al procesar el archivo {audio_path}: {e}")
        return [None]

# MFCC ------------------------------------------------------------------------
def extract_mfcc(y, sr):
    try:
        mfcc= librosa.feature.mfcc(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return mfcc
    except Exception as e:
        print(f"mfcc Error al procesar el archivo {audio_path}: {e}")
        return [None]

# MFCC ------------------------------------------------------------------------
def extract_rms(y, sr):
    try:
        rms= librosa.feature.rms(y=y)
        # print(f"Procesado: {audio_path}")
        return rms
    except Exception as e:
        print(f"rms Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_centroid --------------------------------------------------------------

def extract_spectral_centroid(y, sr):
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_centroid
    except Exception as e:
        print(f"spectral_centroid Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_bandwidth ------------------------------------------------------------
def extract_spectral_bandwidth(y, sr):
    try:
        spectral_bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_bandwidth
    except Exception as e:
        print(f"spectral_bandwidth Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_contrast --------------------------------------------------------------

def extract_spectral_contrast(y, sr):
    try:
        # S = np.abs(librosa.stft(y))
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_contrast
    except Exception as e:
        print(f"spectral_contrast Error al procesar el archivo {audio_path}: {e}")
        return [None]


# spectral_flatness ---------------------------------------------------------------------

def extract_spectral_flatness(y, sr):
    try:
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        # print(f"Procesado: {audio_path}")
        return spectral_flatness
    except Exception as e:
        print(f"spectral_flatness Error al procesar el archivo {audio_path}: {e}")
        return [None]

# spectral_rollof ---------------------------------------------------------------------

def extract_spectral_rolloff(y, sr):
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return spectral_rolloff
    except Exception as e:
        print(f"spectral_rolloff Error al procesar el archivo {audio_path}: {e}")
        return [None]

# poly_features ---------------------------------------------------------------------

def extract_poly_features(y, sr):
    try:
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return poly_features
    except Exception as e:
        print(f"poly_features Error al procesar el archivo {audio_path}: {e}")
        return [None]

# tonnetz ---------------------------------------------------------------------

def extract_tonnetz(y, sr):
    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # print(f"Procesado: {audio_path}")
        return tonnetz
    except Exception as e:
        print(f"tonnetz Error al procesar el archivo {audio_path}: {e}")
        return [None]

# zero_crossing_rate ---------------------------------------------------------------------
def extract_zero_crossing_rate(y, sr):
    try:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        # print(f"Procesado: {audio_path}")
        return zero_crossing_rate
    except Exception as e:
        print(f"zero_crossing_rate Error al procesar el archivo {audio_path}: {e}")
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

    # print(f"Procesado: {audio_path}")
    return (chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc,
     rms, spectral_centroid, spectral_bandwidth, spectral_contrast,
     spectral_flatness, spectral_rolloff, poly_features, tonnetz,
     zero_crossing_rate)


# ------------------------------------------------------------------------------------
#                                 RYTHM FEATURES
# ------------------------------------------------------------------------------------

# tempo ---------------------------------------------------------------------
def extract_tempo(y, sr):
    try:
        # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        tempo = librosa.feature.tempo(y=y, sr=sr)
        return tempo
    except Exception as e:
        print(f"Tempo Error al procesar el archivo: {e}")
        return [None]


# tempogram ---------------------------------------------------------------------
def extract_tempogram(y, sr):
    try:
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        return tempogram
    except Exception as e:
        print(f"Tempogram Error al procesar el archivo: {e}")
        return [None]

    # fourier_tempogram ---------------------------------------------------------------------


def extract_fourier_tempogram(y, sr):
    try:
        fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr)
        return fourier_tempogram
    except Exception as e:
        print(f"Fourier Tempogram Error al procesar el archivo: {e}")
        return [None]

    # tempogram_ratio ---------------------------------------------------------------------


def extract_tempogram_ratio(y, sr):
    try:
        tempogram_ratio = librosa.feature.tempogram_ratio(y=y, sr=sr)
        return tempogram_ratio
    except Exception as e:
        print(f"Tempogram Ratio Error al procesar el archivo: {e}")
        return [None]

    # ALL RYTHM FEATURES-----------------------------------------------------------


def extract_rythm_features(audio_path):
    y, sr = librosa.load(audio_path)

    tempo = extract_tempo(y, sr)
    tempogram = extract_tempogram(y, sr)
    fourier_tempogram = extract_fourier_tempogram(y, sr)
    tempogram_ratio = extract_tempogram_ratio(y, sr)

    return (tempo, tempogram, fourier_tempogram, tempogram_ratio)

#---------------------------------------------------------------------------------

input_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv"


# Leer el archivo CSV original
data = pd.read_csv(input_csv_path)

all_data = []

for index, row in data.iterrows():
    audio_path = row["path"]
    id_audio = row["id_audio"]
    covid_status = row["covid_status"]

    print(f"Procesando...")

    (chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc,
     rms, spectral_centroid, spectral_bandwidth, spectral_contrast,
     spectral_flatness, spectral_rolloff, poly_features, tonnetz,
     zero_crossing_rate) = extract_spectral_features(audio_path)

    tempo, tempogram, fourier_tempogram, tempogram_ratio = extract_rythm_features(audio_path)

    print(f"Procesado: {id_audio}")

    new_row = {
        "id_audio": id_audio,
        #"covid_status": covid_status,

        "chroma_stft": chroma_stft,
        "chroma_cqt": chroma_cqt,
        "chroma_cens": chroma_cens,
        "chroma_vqt": chroma_vqt,
        "melspectrogram": melspectrogram,
        "mfcc": mfcc,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_contrast": spectral_contrast,
        "spectral_flatness": spectral_flatness,
        "spectral_rolloff": spectral_rolloff,
        "poly_features": poly_features,
        "tonnetz": tonnetz,
        "zero_crossing_rate": zero_crossing_rate,

        "tempo": tempo,
        "tempogram": tempogram,
        "fourier_tempogram": fourier_tempogram,
        "tempogram_ratio": tempogram_ratio
    }
    all_data.append(new_row)

# Crear df
feature_df = pd.DataFrame(all_data)

# Guardar el df
output_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/features2.csv"
feature_df.to_csv(output_csv_path, index=False)

