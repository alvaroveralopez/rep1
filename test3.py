import pandas as pd
import numpy as np
import librosa
import os

def extract_tempogram_ratio(y, sr):
    try:
        tempogram_ratio = librosa.feature.tempogram_ratio(y=y, sr=sr)
        return tempogram_ratio
    except Exception as e:
        print(f"Tempogram Ratio Error al procesar el archivo: {e}")
        return None

input_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv"


output_folder = "C:/Users/alvar/PycharmProjects/Proyecto2/features_npy/tempogram_ratio"
# Get the directory path of the current script
#current_directory = os.path.dirname(os.path.abspath(__file__))
#output_folder = os.path.join(current_directory, "features_npy", "tempogram_ratio")

# Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

data = pd.read_csv(input_csv_path)

all_data = []

for index, row in data.iterrows():
    audio_path = row["path"]
    id_audio = os.path.splitext(row["id_audio"])[0]  # Remove the ".wav" extension

    print(f"Procesando: {audio_path}")

    y, sr = librosa.load(audio_path)
    tempogram_ratio = extract_tempogram_ratio(y, sr)

    print(f"Procesado correctamente: {audio_path}")

    # Construct the file path using the output_folder variable and the modified id_audio
    tempogram_ratio_file = f"{output_folder}/{id_audio}.npy"
    os.makedirs(os.path.dirname(tempogram_ratio_file), exist_ok=True)
    np.save(tempogram_ratio_file, tempogram_ratio)

    new_row = {
            "id_audio": id_audio,
            "tempogram_ratio_file": tempogram_ratio_file  # Store the file path
    }
    all_data.append(new_row)


feature_df = pd.DataFrame(all_data)

output_csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/test3.csv"
feature_df.to_csv(output_csv_path, index=False)

