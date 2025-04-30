import pandas as pd
from scipy.stats import normaltest

# Archivos
archivo_entrada_RETIRO = "datos/05precipitaciones_RETIRO_2019_2024.csv"
archivo_entrada_kriging = "datos/06precipdiaria_krigingRETIRO_2019_2024.csv"

# Cargar datos
df_retiro = pd.read_csv(archivo_entrada_RETIRO, sep=";")
d_kr = pd.read_csv(archivo_entrada_kriging)

# Aplicar directamente el test de normalidad
stat, p = normaltest(d_kr["PRECIP_kriging"])
print(f"D’Agostino Test KRIGING: Estadístico={stat:.3f}, p-valor={p:.3f}")

stat, p = normaltest(df_retiro["PRECIPITACION"])
print(f"D’Agostino Test RETIRO: Estadístico={stat:.3f}, p-valor={p:.3f}")
