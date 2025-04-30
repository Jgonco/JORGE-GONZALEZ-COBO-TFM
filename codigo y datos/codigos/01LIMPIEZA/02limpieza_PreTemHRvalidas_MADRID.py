from functools import reduce
import pandas as pd
from itertools import product
import numpy as np

archivo_brutos = "datos/00brutosmeteo_horaria_MADRID_2019_2024.csv"
archivo_salida_ancho = "datos/01meteo_horaria_Madrid_ancho.csv"
archivo_salida_largo = "datos/01meteo_horaria_Madrid_largo.csv"
archivo_estaciones_validas = "datos/Estaciones_validas.csv"

# Cargar datos
validas = pd.read_csv(archivo_estaciones_validas)
ids_validos = validas["ESTACION"].unique()
df_original = pd.read_csv(archivo_brutos, sep=';', encoding='latin1')
df_original = df_original[df_original["ESTACION"].isin(ids_validos)]
for i in range(1, 25):
    h_col = f'H{str(i).zfill(2)}'
    v_col = f'V{str(i).zfill(2)}'
    df_original.loc[df_original[v_col] == 'N', h_col] = np.nan
df_estaciones = validas.copy()

# Filtrado y formato ancho
magnitudes = {83: 'T', 86: 'H', 89: 'P'}
df_filtrado = df_original[df_original['MAGNITUD'].isin(magnitudes.keys())]

columnas_horas = [f'H{str(i).zfill(2)}' for i in range(1, 25)]
columnas_validez = [f'V{str(i).zfill(2)}' for i in range(1, 25)]

dfs = []
for mag, prefix in magnitudes.items():
    df_var = df_filtrado[df_filtrado['MAGNITUD'] == mag].copy()
    for col in columnas_horas:
        df_var.rename(columns={col: f"{prefix}_{col}"}, inplace=True)
    for col in columnas_validez:
        df_var.drop(columns=col, inplace=True)
    df_var = df_var.drop(
        columns=['PROVINCIA', 'MUNICIPIO', 'MAGNITUD', 'PUNTO_MUESTREO'])
    dfs.append(df_var)

df_merged = reduce(lambda left, right: pd.merge(
    left, right, on=["ESTACION", "ANO", "MES", "DIA"], how="outer"), dfs)

# Relleno días faltantes
estaciones = df_merged["ESTACION"].unique()
anos = range(2019, 2025)
meses = range(1, 13)
dias_por_mes = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

fechas_esperadas = pd.DataFrame([
    (est, a, m, d) for est, a, m in product(estaciones, anos, meses)
    for d in range(1, dias_por_mes[m] + 1)
], columns=["ESTACION", "ANO", "MES", "DIA"])

faltantes = fechas_esperadas.merge(
    df_merged[["ESTACION", "ANO", "MES", "DIA"]].drop_duplicates(),
    on=["ESTACION", "ANO", "MES", "DIA"], how="left", indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])

for var in magnitudes.values():
    for col in columnas_horas:
        faltantes[f"{var}_{col}"] = None

df_completo = pd.concat([df_merged, faltantes], ignore_index=True)
df_completo = df_completo.sort_values(
    by=["ESTACION", "ANO", "MES", "DIA"]).reset_index(drop=True)

# Interpolación temporal


def interpolar_variable(df, var_prefix):
    for j in range(24):
        colum = f"{var_prefix}_H{str(j+1).zfill(2)}"
        df[colum] = pd.to_numeric(df[colum], errors='coerce')

        for idx in range(1, len(df) - 1):
            if pd.isna(df.at[idx, colum]):
                prev_row = df.iloc[idx - 1]
                next_row = df.iloc[idx + 1]

                if (
                    prev_row['ESTACION'] == df.at[idx, 'ESTACION']
                    and next_row['ESTACION'] == df.at[idx, 'ESTACION']
                    and not pd.isna(prev_row[colum])
                    and not pd.isna(next_row[colum])
                ):
                    df.at[idx, colum] = (prev_row[colum] + next_row[colum]) / 2
    return df


for var in magnitudes.values():
    df_completo = interpolar_variable(df_completo, var)

# Interpolación espacial IDW
df_coords = df_estaciones[["ESTACION", "LONGITUD", "LATITUD"]]
df_completo = df_completo.merge(df_coords, on="ESTACION", how="left")


def aplicar_idw_mixto(df, var_prefix, power=2, alpha=0.5):
    for k in range(24):
        columna = f"{var_prefix}_H{str(k+1).zfill(2)}"
        for idx in df.index[df[columna].isna()]:
            est = df.at[idx, 'ESTACION']
            lon, lat = df.at[idx, 'LONGITUD'], df.at[idx, 'LATITUD']
            same_time = (
                (df["ANO"] == df.at[idx, "ANO"]) &
                (df["MES"] == df.at[idx, "MES"]) &
                (df["DIA"] == df.at[idx, "DIA"]) &
                (~df[columna].isna()) &
                (df["ESTACION"] != est)
            )
            vecinos = df[same_time]
            vecinos = vecinos[vecinos["ESTACION"].isin(ids_validos)]
            if vecinos.empty:
                continue

            dx = vecinos["LONGITUD"].values - lon
            dy = vecinos["LATITUD"].values - lat
            dist = np.sqrt(dx**2 + dy**2)
            weights = 1.0 / (dist**power + 1)  # suavizado de peso
            valores = vecinos[columna].values

            idw_val = np.sum(weights * valores) / np.sum(weights)
            media_val = np.mean(valores)
            df.at[idx, columna] = alpha * idw_val + (1 - alpha) * media_val
    return df


for var in ['T', 'H', 'P']:
    df_completo = aplicar_idw_mixto(df_completo, var)

# Formato ancho
df_completo = df_completo.drop(columns=["LONGITUD", "LATITUD"])
df_completo = df_completo.rename(
    columns={"ANO": "year", "MES": "month", "DIA": "day"})
df_completo["FECHA"] = pd.to_datetime(
    df_completo[["year", "month", "day"]], errors='coerce')
df_completo = df_completo.dropna(subset=["FECHA"]).drop(
    columns=["year", "month", "day"])

df_completo.iloc[:, 2:-1] = df_completo.iloc[:, 2:-
                                             1].applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

df_completo.to_csv(archivo_salida_ancho, index=False,
                   sep=';', encoding='latin1')
print(f"✔ Archivo ANCHO guardado: {archivo_salida_ancho}")

# Formato largos
df_largo = pd.DataFrame()
for i in range(1, 25):
    h = f'H{str(i).zfill(2)}'
    temp = df_completo[[f'T_{h}']].rename(columns={f'T_{h}': 'TEMPERATURA'})
    hum = df_completo[[f'H_{h}']].rename(columns={f'H_{h}': 'HUMEDADR'})
    prec = df_completo[[f'P_{h}']].rename(columns={f'P_{h}': 'PRECIPITACION'})

    bloque = df_completo[['ESTACION', 'FECHA']].copy()
    bloque['FECHA'] = (bloque['FECHA'] + pd.to_timedelta(i -
                       1, unit='h')).dt.strftime('%Y-%m-%d %H')
    bloque['TEMPERATURA'] = temp.values.round(2)
    bloque['HUMEDADR'] = hum.values.round(2)
    bloque['PRECIPITACION'] = prec.values.round(2)

    df_largo = pd.concat([df_largo, bloque], ignore_index=True)

df_largo = df_largo[['ESTACION', 'FECHA',
                     'PRECIPITACION', 'TEMPERATURA', 'HUMEDADR']]
df_largo = df_largo.sort_values(
    by=["ESTACION", "FECHA"]).reset_index(drop=True)

df_largo.to_csv(archivo_salida_largo, index=False, sep=';', encoding='latin1')
print(f" Archivo LARGO guardado: {archivo_salida_largo}")
