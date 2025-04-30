import pandas as pd

# Cargar el CSV
archivo_entrada = "datos/10brutoaccidentes_madrid.csv"
archivo_salida = "datos/11accidentes_Madrid.csv"


df_acc = pd.read_csv(archivo_entrada, sep=";")

# Creación la nueva columna de fecha y hora, truncada a la hora completa
df_acc["FECHA"] = pd.to_datetime(df_acc["fecha"] + " " + df_acc["hora"])
df_acc["FECHA"] = df_acc["FECHA"].dt.floor("H")

# Eliminación columnas innecesarias y duplicados
df_acc = df_acc.drop(columns=["fecha", "hora", "localizacion"])

df_acc = df_acc.drop_duplicates()


df_acc = df_acc[["FECHA", "coordenada_x_utm", "coordenada_y_utm"]]

# Guardar
df_acc.to_csv(archivo_salida, sep=",", index=False)
