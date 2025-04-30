import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Archivos
archivo_entrada_RET = "datos/07precipmens_RETIRO_2019_2024.csv"
archivo_entrada_KR = "datos/08precipmens_krigingRETIRO_2019_2024.csv"

# Cargar datos
df_mens_RET = pd.read_csv(archivo_entrada_RET)
df_mens_KR = pd.read_csv(archivo_entrada_KR)

# Convertir FECHA a datetime
df_mens_RET["FECHA"] = pd.to_datetime(df_mens_RET["FECHA"], format="%Y-%m")
df_mens_KR["FECHA"] = pd.to_datetime(df_mens_KR["FECHA"], format="%Y-%m")

# Crear figura
plt.figure(figsize=(12, 6))

# Plot RETIRO
plt.plot(df_mens_RET["FECHA"], df_mens_RET["PRECIPmes"], marker="o",
         linestyle="-", label="PRECIPITACIÓN RETIRO MENSUAL", color="blue")

# Plot KRIGING
plt.plot(df_mens_KR["FECHA"], df_mens_KR["PRECIPmes"], marker="s",
         linestyle="--", label="PRECIPITACIÓN KRIGING MENSUAL", color="red")

# Personalización del gráfico
plt.xlabel("Fecha")
plt.ylabel("Precipitación mensual (mm)")
plt.title("Comparación de Precipitaciones (2019-2024)")
plt.legend()
plt.grid(True)

# Formato de fechas en el eje x
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Rotar etiquetas
plt.xticks(rotation=45)

# Mostrar gráfico
plt.show()
