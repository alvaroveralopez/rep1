import os
import pandas as pd
from tabulate import tabulate


# Directorio principal
main_directory = "C:/Users/alvar/PycharmProjects/Proyecto2/Coswara-Data-master"


subdirectories = [d for d in os.listdir(main_directory) if
                  os.path.isdir(f"{main_directory}/{d}") and d.startswith("202")]

final_data = []

# Recorrer las carpetas
for subdir in subdirectories:
    # Ruta al csv del paciente
    csv_path = f"{main_directory}/{subdir}/{subdir}.csv"

    df = pd.read_csv(csv_path) # dataframe con los audios del paciente

    # Recorrer el df del paciente
    for index, row in df.iterrows():

        paciente = row["id"]
        audios_dir = f"{main_directory}/Extracted_data/{subdir}/{paciente}"

        # Recorrer los audios de cada paciente
        audio_count = 0
        for audio_file in os.listdir(audios_dir):
            if audio_file.endswith(".wav"):
                audio_count += 1

                audio_type = os.path.splitext(audio_file)[0]
                id_audio = f"{subdir}/{paciente}/{audio_file}"
                new_row = {
                    "id_audio": id_audio,
                    "id_patient": paciente,
                    "audio_type": audio_type,
                    "covid_status": row["covid_status"],
                    "age": row["a"],
                    "gender": row["g"],
                    "path": f"{audios_dir}/{audio_file}"
                }
                # Añadir la fila al df
                final_data.append(new_row)

        if audio_count != 9:
            print(f"Paciente {audios_dir} tiene {audio_count} audios")

# DataFrame con los datos finales
final_df = pd.DataFrame(final_data)

# Guardar como CSV
final_csv_path = "global_data.csv"
final_df.to_csv(final_csv_path, index=False)

# csv_file_path = 'C:/Users/alvar/PycharmProjects/Proyecto2/global_data.csv'

#data_frame = pd.read_csv(csv_file_path)
#print(data_frame)


print(final_df.head(25))
print(tabulate(final_df, headers='keys', tablefmt='psql'))

