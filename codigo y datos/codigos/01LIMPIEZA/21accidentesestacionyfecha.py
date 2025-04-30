import pandas as pd


archivo_entrada = "datos/20accidentesyprecipitacion.csv"
archivo_salida = "datos/21accidentesporestacion.csv"
# Leer CSV original
df_accyprec = pd.read_csv(archivo_entrada, parse_dates=["FECHA"])

# Redondear FECHA a la hora
df_accyprec["FECHA"] = df_accyprec["FECHA"].dt.floor("H")

# Contar accidentes por ESTACION, FECHA y PRECIPITACION
df_accyprec["ACCIDENTES"] = 1
df_agrupado = df_accyprec.groupby(["ESTACION", "FECHA", "PRECIPITACION"]).agg(
    {"ACCIDENTES": "sum"}).reset_index()

# Crear rango de fechas por hora
rango_fechas = pd.date_range(
    start="2019-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")

# Obtener lista única de estaciones
estaciones = df_agrupado["ESTACION"].unique()

# Crear un DataFrame completo con todas las combinaciones
completo = pd.MultiIndex.from_product(
    [estaciones, rango_fechas], names=["ESTACION", "FECHA"]
).to_frame(index=False)

# Hacer merge para rellenar datos reales
df_accyprec_est = pd.merge(completo, df_agrupado, on=[
                           "ESTACION", "FECHA"], how="left")
df_accyprec_est["ACCIDENTES"] = df_accyprec_est["ACCIDENTES"].fillna(
    0).astype(int)


df_accyprec_est = df_accyprec_est.sort_values(["ESTACION", "FECHA"])

# Guardar
df_accyprec_est.to_csv(archivo_salida, index=False)
