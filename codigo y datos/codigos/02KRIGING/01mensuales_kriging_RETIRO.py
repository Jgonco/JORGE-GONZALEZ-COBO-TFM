import pandas as pd


archivo_entrada_RETIRO = "datos/05precipitaciones_RETIRO_2019_2024.csv"
archivo_entrada_kriging = "datos/06precipdiaria_krigingRETIRO_2019_2024.csv"
archivo_salida_RETIRO = "datos/07precipmens_RETIRO_2019_2024.csv"
archivo_salida_kriging = "datos/08precipmens_krigingRETIRO_2019_2024.csv"

df_RET = pd.read_csv(archivo_entrada_RETIRO, sep=";")

df_kr = pd.read_csv(archivo_entrada_kriging)


df_RET['FECHA'] = pd.to_datetime(df_RET['FECHA'])

df_kr['FECHA'] = pd.to_datetime(df_kr['FECHA'])

# Agrupar por ESTACION y por mes RETIRO
df_RET_mensual = df_RET.groupby(["ESTACION", pd.Grouper(
    key="FECHA", freq="M")], as_index=False)["PRECIPITACION"].sum().round(1)
df_RET_mensual["FECHA"] = df_RET_mensual["FECHA"].dt.strftime('%Y-%m')
# Renombrar la columna de precipitaciones
df_RET_mensual.rename(columns={"PRECIPITACION": "PRECIPmes"}, inplace=True)


# Agrupar por ESTACION y por mes KRIGING
df_kr_mensual = df_kr.groupby(["ESTACION", pd.Grouper(
    key="FECHA", freq="M")], as_index=False)["PRECIP_kriging"].sum().round(1)
df_kr_mensual["FECHA"] = df_kr_mensual["FECHA"].dt.strftime('%Y-%m')
# Renombrar la columna de precipitaciones
df_kr_mensual.rename(columns={"PRECIP_kriging": "PRECIPmes"}, inplace=True)


df_RET_mensual.to_csv(archivo_salida_RETIRO, index=False)
df_kr_mensual.to_csv(archivo_salida_kriging, index=False)
