import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
np.set_printoptions(threshold=np.inf)
from tabulate import tabulate
# import json
# import pickle


# Ruta al archivo CSV
csv_path = "C:/Users/alvar/PycharmProjects/Proyecto2/features.csv"

loaded_df = pd.read_csv(csv_path)


m_tempo = np.load(loaded_df.iloc[19459]["tempo"])
m_tempogram = np.load(loaded_df.iloc[19459]["tempogram"])
m_fourier_tempogram = np.load(loaded_df.iloc[19459]["fourier_tempogram"])
m_tempogram_ratio = np.load(loaded_df.iloc[19459]["tempogram_ratio"])

print(f"m_tempogram: {m_tempogram}")
print(f"Size of m_tempo: {m_tempo.shape}")
print(f"Size of m_tempogram: {m_tempogram.shape}")
print(f"Size of m_fourier_tempogram: {m_fourier_tempogram.shape}")
print(f"Size of m_tempogram_ratio: {m_tempogram_ratio.shape}")

# print(np.load(loaded_df.iloc[1]["tempogram_ratio"]))


# print(data.head())
print(tabulate(loaded_df.head(), headers='keys', tablefmt='psql'))
#print(data.tail())
print(tabulate(loaded_df.tail(), headers='keys', tablefmt='psql'))


num_rows = loaded_df.shape[0]
print(f"Número de filas en el CSV: {num_rows}")
