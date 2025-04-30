import pandas as pd


archivo_entrada = "datos/00brutosmeteo_horaria_MADRID_2019_2024.csv"
archivo_salida = "datos/00brutosprecip_horaria_MADRID.csv"

# Cargar archivo original
df = pd.read_csv(archivo_entrada, sep=";", encoding="latin1")

# Filtrar solo MAGNITUD 89 (precipitación)
df_precip = df[df["MAGNITUD"] == 89].copy()

# Listas de columnas H y V
horas = [f"H{str(i).zfill(2)}" for i in range(1, 25)]
validaciones = [f"V{str(i).zfill(2)}" for i in range(1, 25)]

# Eliminar valores no válidos (donde Vxx == 'N')
for h_col, v_col in zip(horas, validaciones):
    df_precip.loc[df_precip[v_col] == "N", h_col] = None

# Convertir valores a float
for h_col in horas:
    df_precip[h_col] = pd.to_numeric(df_precip[h_col].astype(
        str).str.replace(",", "."), errors="coerce")

# Seleccionar columnas necesarias
df_precip = df_precip[["ESTACION", "ANO", "MES", "DIA"] + horas]

# Crear columna FECHA
df_precip["FECHA"] = pd.to_datetime(
    df_precip[["ANO", "MES", "DIA"]].rename(
        columns={"ANO": "year", "MES": "month", "DIA": "day"})
)

# Crear DataFrame con todas las fechas del rango
rango_fechas = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")

# Obtener lista única de estaciones
estaciones = df_precip["ESTACION"].unique()

# Crear todas las combinaciones posibles ESTACION x FECHA
completo = pd.MultiIndex.from_product([estaciones, rango_fechas], names=[
                                      "ESTACION", "FECHA"]).to_frame(index=False)

# Unir con los datos existentes
df_completo = pd.merge(completo, df_precip, on=[
                       "ESTACION", "FECHA"], how="left")

# Rellenar columnas de fecha descompuesta
df_completo["ANO"] = df_completo["FECHA"].dt.year
df_completo["MES"] = df_completo["FECHA"].dt.month
df_completo["DIA"] = df_completo["FECHA"].dt.day

# Reordenar columnas
columnas_finales = ["ESTACION", "ANO", "MES", "DIA"] + horas
df_resultado = df_completo[columnas_finales]

# Ordenar por estación y fecha
df_resultado = df_resultado.sort_values(by=["ESTACION", "ANO", "MES", "DIA"])

# Guardar resultado
df_resultado.to_csv(archivo_salida, sep=";", index=False, encoding="latin1")
