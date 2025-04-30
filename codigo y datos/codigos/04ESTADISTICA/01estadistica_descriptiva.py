import pandas as pd

archivo_entrada_PRECIP = "datos/01meteo_horaria_Madrid_largo.csv"
archivo_entrada_accidentes = "datos/21accidentesporestacion.csv"
archivo_salida_estadisPRECIP = "datos/estadisticas_precipitacion_estacion_mes_anio.csv"
archivo_salida_estadisACCIDENTES = "datos/estadisticas_accidentes_estacion_mes_anio.csv"


# PRECIPITACIONES
df_meteo = pd.read_csv(archivo_entrada_PRECIP, sep=';')
df_meteo.columns = df_meteo.columns.str.strip().str.upper()
df_meteo['FECHA'] = pd.to_datetime(df_meteo['FECHA'], format="%Y-%m-%d %H")
df_meteo['AÑO'] = df_meteo['FECHA'].dt.year
df_meteo['MES'] = df_meteo['FECHA'].dt.month

# Estadísticas PRECIPITACIÓN
estadisticas_precip = df_meteo.groupby(['ESTACION', 'AÑO', 'MES'])['PRECIPITACION'].agg(
    count='count',
    mean='mean',
    std='std',
    min='min',
    q25=lambda x: x.quantile(0.25),
    q50='median',
    q75=lambda x: x.quantile(0.75),
    max='max'
).reset_index().round(2)

# Renombrar columnas
estadisticas_precip.columns = ['ESTACION', 'AÑO', 'MES', 'count',
                               'mean', 'std', 'min', '25%', '50%', '75%', 'max']

# Guardar
estadisticas_precip.to_csv(
    archivo_salida_estadisPRECIP, index=False)

# ACCIDENTES
df_accidentes = pd.read_csv(archivo_entrada_accidentes)
df_accidentes.columns = df_accidentes.columns.str.strip().str.upper()
df_accidentes['FECHA'] = pd.to_datetime(df_accidentes['FECHA'])
df_accidentes['AÑO'] = df_accidentes['FECHA'].dt.year
df_accidentes['MES'] = df_accidentes['FECHA'].dt.month

# Estadísticas ACCIDENTES
estadisticas_accidentes = df_accidentes.groupby(['ESTACION', 'AÑO', 'MES'])['ACCIDENTES'].agg(
    count='count',
    mean='mean',
    std='std',
    min='min',
    q25=lambda x: x.quantile(0.25),
    q50='median',
    q75=lambda x: x.quantile(0.75),
    max='max'
).reset_index().round(2)

# Renombrar columnas
estadisticas_accidentes.columns = ['ESTACION', 'AÑO', 'MES',
                                   'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

# Guardar
estadisticas_accidentes.to_csv(archivo_salida_estadisACCIDENTES, index=False)
