import pandas as pd
import numpy as np
import librosa
import os
from tabulate import tabulate

# ------------------------------------------------------------------------------------
#                                 RYTHM FEATURES
# ------------------------------------------------------------------------------------

# tempo ---------------------------------------------------------------------
def extract_tempo(y, sr):
    try:
        #onset_env = librosa.onset.onset_strength(y=y, sr=sr)
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
output_folder = "C:/Users/alvar/PycharmProjects/Proyecto2/features_npy"



data = pd.read_csv(input_csv_path)

all_data = []


for index, row in data.iterrows():
    audio_path = row["path"]
    covid_status = row["covid_status"]
    id_audio = row["id_audio"]

    print(f"Procesando...")

    tempo, tempogram, fourier_tempogram, tempogram_ratio = extract_rythm_features(audio_path)
    print(f"Procesado: {audio_path}")

    tempo_file = f"{output_folder}/tempo/{id_audio}.npy"
    os.makedirs(os.path.dirname(tempo_file), exist_ok=True)
    np.save(tempo_file, tempo)

    tempogram_file = f"{output_folder}/tempogram/{id_audio}.npy"
    os.makedirs(os.path.dirname(tempogram_file), exist_ok=True)
    np.save(tempogram_file, tempogram_ratio)

    fourier_tempogram_file = f"{output_folder}/fourier_tempogram/{id_audio}.npy"
    os.makedirs(os.path.dirname(fourier_tempogram_file), exist_ok=True)
    np.save(fourier_tempogram_file, tempogram_ratio)

    tempogram_ratio_file = f"{output_folder}/tempogram_ratio/{id_audio}.npy"
    os.makedirs(os.path.dirname(tempogram_ratio_file), exist_ok=True)
    np.save(tempogram_ratio_file, tempogram_ratio)



    new_row = {
        "id_audio": id_audio,
        "tempo": tempo_file,
        "tempogram": tempogram_file,
        "fourier_tempogram": fourier_tempogram_file,
        "tempogram_ratio": tempogram_ratio_file
    }
    all_data.append(new_row)

feature_df = pd.DataFrame(all_data)

output_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/rythm_features.csv"
feature_df.to_csv(output_csv_path, index=False)
