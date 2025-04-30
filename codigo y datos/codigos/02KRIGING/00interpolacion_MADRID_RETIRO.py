from pykrige.uk import UniversalKriging
import numpy as np
import pandas as pd


archivo_precipitaciones_MADRID = "datos/00brutosprecip_horaria_MADRID.csv"
archivo_estaciones = "datos/Estaciones_validas.csv"
archivo_retiro = "datos/Estacion_Retiro.csv"
archivo_salida = "datos/06precipdiaria_krigingRETIRO_2019_2024.csv"


df_horaria_MADRID = pd.read_csv(
    archivo_precipitaciones_MADRID, sep=";", encoding="latin1")


# Suma diaria
columnas_horas = [f"H{str(i).zfill(2)}" for i in range(1, 25)]
df_horaria_MADRID["PRECIPDIARIA"] = df_horaria_MADRID[columnas_horas].astype(
    float).sum(axis=1)


df_horaria_MADRID["FECHA"] = pd.to_datetime(
    df_horaria_MADRID[["ANO", "MES", "DIA"]])

df_precipDia_MADRID = df_horaria_MADRID[["ESTACION", "FECHA", "PRECIPDIARIA"]]

df_estaciones = pd.read_csv(archivo_estaciones)
df_estacion_retiro = pd.read_csv(archivo_retiro)


lon_retiro = df_estacion_retiro["LONGITUD"].values[0]
lat_retiro = df_estacion_retiro["LATITUD"].values[0]


df_precycoor_MADRID = df_precipDia_MADRID.merge(df_estaciones, on="ESTACION")

resultados = []

# iteracion sobre cada dia
for fecha, grupo in df_precycoor_MADRID.groupby("FECHA"):
    x = grupo["LONGITUD"].values
    y = grupo["LATITUD"].values
    precipitaciones_diarias = grupo["PRECIPDIARIA"].values

    # Ruido para evitar errores
    precipitaciones_diarias += np.random.normal(
        0, 0.001, size=precipitaciones_diarias.shape)

    # Kriging
    UK = UniversalKriging(x, y, precipitaciones_diarias,
                          drift_terms=["regional_quadratic"],
                          variogram_model="spherical",
                          verbose=False,
                          enable_plotting=False)
    prec_kriging, _ = UK.execute("points", lon_retiro, lat_retiro)

    # precipitaciones no negativas
    prec_kriging = max(0, round(prec_kriging[0], 1))

    resultados.append([3195, fecha, prec_kriging])


df_kriging_Retiro = pd.DataFrame(
    resultados, columns=["ESTACION", "FECHA", "PRECIP_kriging"])


df_kriging_Retiro.to_csv(archivo_salida, index=False,
                         sep=",", encoding="latin1")
