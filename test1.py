import pandas as pd

'''
df1 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/spectral_features.csv")
df2 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features_CSVs/rythm_features.csv")

df1.rename(columns={'covid_status': 'id_audio'}, inplace=True)

df2 = df2.drop(columns=["id_audio"])

global_df = pd.concat([df1, df2], axis=1)

global_df.to_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features.csv", index=False)
'''

9

df1 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features.csv")
df2 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv")

valor_a_filtrar = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/breathing-deep.wav'
valor_a_filtrar1 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/breathing-shallow.wav'
valor_a_filtrar2 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/cough-heavy.wav'
valor_a_filtrar3 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/cough-shallow.wav'
valor_a_filtrar4 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/counting-fast.wav'
valor_a_filtrar5 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/counting-normal.wav'
valor_a_filtrar6 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/vowel-a.wav'
valor_a_filtrar7 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/vowel-e.wav'
valor_a_filtrar8 = '20200413/9XKj7fAmvwPUas9GFPZuTpev7T03/vowel-o.wav'

# Filtrar las filas donde el valor en la columna "columna_nombre" coincide
df1 = df1[df1['id_audio'] != valor_a_filtrar]
df1 = df1[df1['id_audio'] != valor_a_filtrar1]
df1 = df1[df1['id_audio'] != valor_a_filtrar2]
df1 = df1[df1['id_audio'] != valor_a_filtrar3]
df1 = df1[df1['id_audio'] != valor_a_filtrar4]
df1 = df1[df1['id_audio'] != valor_a_filtrar5]
df1 = df1[df1['id_audio'] != valor_a_filtrar6]
df1 = df1[df1['id_audio'] != valor_a_filtrar7]
df1 = df1[df1['id_audio'] != valor_a_filtrar8]

df2 = df2[df2['id_audio'] != valor_a_filtrar]
df2 = df2[df2['id_audio'] != valor_a_filtrar1]
df2 = df2[df2['id_audio'] != valor_a_filtrar2]
df2 = df2[df2['id_audio'] != valor_a_filtrar3]
df2 = df2[df2['id_audio'] != valor_a_filtrar4]
df2 = df2[df2['id_audio'] != valor_a_filtrar5]
df2 = df2[df2['id_audio'] != valor_a_filtrar6]
df2 = df2[df2['id_audio'] != valor_a_filtrar7]
df2 = df2[df2['id_audio'] != valor_a_filtrar8]

# Guardar los cambios en el mismo archivo CSV
df1.to_csv('features.csv', index=False)
df2.to_csv('global_data.csv', index=False)

'''

df1 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv")
df2 = pd.read_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features.csv")

df_id_audio = df1[["id_audio"]]
df2 = df2.drop(columns=["covid_status"])

global_df = pd.concat([df_id_audio, df2], axis=1)

global_df.to_csv("C:/Users/alvar/PycharmProjects/Proyecto2/features_2.csv", index=False)
'''
