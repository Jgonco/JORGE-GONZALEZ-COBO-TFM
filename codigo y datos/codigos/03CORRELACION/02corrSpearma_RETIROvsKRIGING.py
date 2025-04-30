import pandas as pd
from scipy.stats import spearmanr

archivo_entrada = "datos/05precipitaciones_RETIRO_2019_2024.csv"
# Cargar datos reales (separador ;)
df_RETIRO = pd.read_csv(archivo_entrada, sep=";")

# Cargar datos kriging (separador ,)
d_kr = pd.read_csv("datos/06precipdiaria_krigingRETIRO_2019_2024.csv")

# Convertir columnas FECHA a datetime
df_RETIRO["FECHA"] = pd.to_datetime(df_RETIRO["FECHA"], errors='coerce')
d_kr["FECHA"] = pd.to_datetime(
    d_kr["FECHA"], errors='coerce')

# Unir por ESTACION y FECHA
df_RETyKR = pd.merge(df_RETIRO, d_kr, on=["ESTACION", "FECHA"])

# Calcular correlación de Spearman
coef, p_value = spearmanr(
    df_RETyKR["PRECIPITACION"], df_RETyKR["PRECIP_kriging"])
print(
    f"Coeficiente de correlación Spearman: {coef:.3f}, p-valor: {p_value:.3f}")
