# This is a sample Python script.
import os
import librosa
import numpy as np
from sklearn.linear_model import LinearRegression

def read_audios_from_folder(folder_path: str):
    """
    This function reads all the audios from a folder using librosa and returns a list of them
    :param folder_path:
    :return: dict of audios with the name of the file as key
    """
    list_extensions = (".wav", ".mp3", ".ogg", ".m4a")

    audios_out: dict = {}
    for file in os.listdir(folder_path):
        if file.endswith(list_extensions):
            audio_path = os.path.join(folder_path, file)
            raw_audio, sampling_rate = librosa.load(audio_path)
            print(f"Audio {file} loaded")

            audios_out[file] = [raw_audio, sampling_rate]
    return audios_out


def ext_mfcc(path_in: str):
    """
    This function extracts the mfcc from all the audios stored in a folder
    :param path_in: path of the audios
    :return: a dict with all mfcc as a numpy array and the name of the file as key
    """
    dict_audios: dict = read_audios_from_folder(path_in)
    dict_mfcc: dict = {}
    for key, value in dict_audios.items():
        audio_mfcc: np.array = librosa.feature.mfcc(y=value[0], sr=value[1])
        dict_mfcc[key] = audio_mfcc
        print(f"Audio {key} processed")

    return dict_mfcc


def make_labels_at_frame_leve(dict_mfcc: dict):
    """
    Generates the labels for the frames of the mfcc
    :param dict_mfcc: a dict with the matrix of mfcc for each audio and the name of the file as key
    :return: a dict with the labels for each frame of the mfcc
    """
    dict_labels: dict = {}
    for key, value in dict_mfcc.items():
        labels = np.zeros(value.shape[1])
        dict_labels[key] = labels
    return dict_labels


def make_dataset(dict_mfcc: dict, dict_labels: dict):
    all_mfcc = np.empty((20, 0))
    all_labels = np.empty(1)
    for key, value in dict_mfcc.items():
        mfcc = value
        labels = dict_labels[key]

        all_mfcc = np.concatenate((all_mfcc, mfcc), axis=1)
        all_labels = np.concatenate((all_labels, labels))

    return all_mfcc, all_labels[1:]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "C:\\Users\\admin_visia\\Downloads"
    feats_mfcc = ext_mfcc(path)
    labels = make_labels_at_frame_leve(feats_mfcc)

    # Train set of mfcc and labels
    X_mfcc, y_labels = make_dataset(feats_mfcc, labels)

    # Creando un clasificador basado en Regresión Lineal
    model = LinearRegression()

    # Train the model using the training sets
    model.fit(X_mfcc.T, y_labels)

    # Predict the labels for the training set
    frame = X_mfcc.T[0].reshape(1, -1)
    y_pred = model.predict(frame)

    print("Done!")
