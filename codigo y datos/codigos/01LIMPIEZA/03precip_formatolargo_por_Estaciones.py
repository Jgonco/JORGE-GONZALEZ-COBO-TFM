import pandas as pd


archivo_entrada = "datos/01meteo_horaria_Madrid_largo.csv"
archivo_salida_precip = "datos/02precip_horaria_Madrid_largo_estaciones.csv"
archivo_salida_tem = "datos/02temperatura_horaria_Madrid_largo_estaciones.csv"
archivo_salida_hum = "datos/02humedad_horaria_Madrid_largo_estaciones.csv"

# Cargar archivo original
df_meteo = pd.read_csv(archivo_entrada, sep=";", encoding="utf-8")

# Asegurar que la columna FECHA es datetime
df_meteo["FECHA"] = pd.to_datetime(df_meteo["FECHA"])

# Filtrar solo precipitaciones (nos quedamos con ESTACION, FECHA y PRECIPITACION)
df_precip = df_meteo[["ESTACION", "FECHA", "PRECIPITACION"]].copy()
df_temp = df_meteo[["ESTACION", "FECHA", "TEMPERATURA"]].copy()
df_hum = df_meteo[["ESTACION", "FECHA", "HUMEDADR"]].copy()

# PRECIPITACION
# Pivotear: cada estación será una columna
df_pivot_prec = df_precip.pivot_table(
    index="FECHA", columns="ESTACION", values="PRECIPITACION")

# Renombrar columnas agregando prefijo S
df_pivot_prec.columns = [f"S{int(col)}" for col in df_pivot_prec.columns]


# Reset index para que FECHA sea una columna
df_pivot_prec = df_pivot_prec.reset_index()

# Guardar resultado
df_pivot_prec.to_csv(archivo_salida_precip, index=False,
                     sep=",", encoding="utf-8")

# TEMPERATURA
# Pivotear: cada estación será una columna
df_pivot_temp = df_temp.pivot_table(
    index="FECHA", columns="ESTACION", values="TEMPERATURA")

# Renombrar columnas agregando prefijo S
df_pivot_temp.columns = [f"S{int(col)}" for col in df_pivot_temp.columns]

# Reset index para que FECHA sea una columna
df_pivot_temp = df_pivot_temp.reset_index()

# Guardar resultado
df_pivot_temp.to_csv(archivo_salida_tem, index=False,
                     sep=",", encoding="utf-8")

# HUMEDAD
# Pivotear: cada estación será una columna
df_pivot_hum = df_hum.pivot_table(
    index="FECHA", columns="ESTACION", values="HUMEDADR")

# Renombrar columnas agregando prefijo S
df_pivot_hum.columns = [f"S{int(col)}" for col in df_pivot_hum.columns]

# Reset index para que FECHA sea una columna
df_pivot_hum = df_pivot_hum.reset_index()

# Guardar resultado
df_pivot_hum.to_csv(archivo_salida_hum, index=False, sep=",", encoding="utf-8")
