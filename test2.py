import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import ast

# Cargar el CSV como DataFrame
csv_path = "features_2.csv"
dataframe = pd.read_csv(csv_path)


numpy_array = dataframe.iloc[1, "chroma_stft"].values


# Mostrar el elemento y su forma (shape)
print("Elemento:")
print(numpy_array)
print("Forma del elemento:", numpy_array.shape)