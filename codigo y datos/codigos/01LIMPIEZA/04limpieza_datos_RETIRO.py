import pandas as pd

archivo_entrada = "datos/03brutosmeteo_diaria_RETIRO_2019_2024.csv"
archivo_salida = "datos/05precipitaciones_RETIRO_2019_2024.csv"
col = ['indicativo', 'fecha', 'prec']
df_retirobruto = pd.read_csv(
    archivo_entrada, sep=';', encoding='utf-8', usecols=col)

df_retirobruto['FECHA'] = pd.to_datetime(df_retirobruto['fecha'])

df_retirobruto = df_retirobruto.rename(columns={'indicativo': 'ESTACION'})

df_retirobruto.columns = df_retirobruto.columns.str.strip()

df_retirobruto['prec'] = df_retirobruto['prec'].replace("Ip", pd.NA)

df_retirobruto['prec'] = df_retirobruto['prec'].replace({',': '.'}, regex=True)

df_retirobruto['prec'] = pd.to_numeric(df_retirobruto['prec'], errors='coerce')

df_retirobruto.sort_values(by=['FECHA'], inplace=True)

# Reemplazar los valores NaN con la media del día anterior y siguiente
for i in range(1, len(df_retirobruto)-1):
    if pd.isna(df_retirobruto.loc[i, 'prec']):

        anterior = df_retirobruto.loc[i-1, 'prec']
        posterior = df_retirobruto.loc[i+1, 'prec']

        if not pd.isna(anterior) and not pd.isna(posterior):
            df_retirobruto.loc[i, 'prec'] = (anterior + posterior) / 2

        elif pd.isna(anterior) or pd.isna(posterior):
            df_retirobruto.loc[i, 'prec'] = pd.NA

df_retirolimpio = df_retirobruto.drop_duplicates()

# Renombrar prec
df_retirolimpio = df_retirolimpio.rename(columns={'prec': 'PRECIPITACION'})

# Seleccionar las columnas en el orden deseado
df_precipDia_Retiro = df_retirolimpio[['ESTACION', 'FECHA', 'PRECIPITACION']]

df_precipDia_Retiro.to_csv(archivo_salida,
                           index=False, sep=";")
