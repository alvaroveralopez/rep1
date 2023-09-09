import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np


def cargar_metadatos(ruta_csv):
    # Cargar los datos en un DataFrame
    df = pd.read_csv(ruta_csv)

    return df

def cargar_audio(ruta_audio):
    audio, sr = librosa.load(ruta_audio, sr=None)
    return audio, sr

if __name__ == "__main__":
    # Ruta del archivo CSV con los metadatos
    ruta_csv = "20200413/20200413.csv"
    carpeta_audios = "20200413/20200413"

    # Cargar los metadatos
    metadatos = cargar_metadatos(ruta_csv)
    # print(metadatos.head())

    # Edad (a) y género (g)
    filtro_edad = metadatos["a"] >= 30
    filtro_genero = metadatos["g"] == "female"

    sel_audios = metadatos[filtro_edad & filtro_genero]
    espectrogramas = []

    figuras = []

    for indice, fila in sel_audios.iterrows():
        id_audio = fila["id"]

        ruta_audio = carpeta_audios + "/" + id_audio + "/breathing-deep.wav"
        audio, sr = cargar_audio(ruta_audio)

        # Extraer el espectrograma
        espectrograma = librosa.feature.melspectrogram(y=audio, sr=sr)
        espectrogramas.append(espectrograma)

        # Representar
        fig, ax = plt.subplots()

        librosa.display.specshow(librosa.power_to_db(espectrograma, ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Espectrograma del audio {id_audio}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')

        figuras.append(fig)


    print("Cantidad de audios cargados:", len(espectrogramas))

    plt.show()

