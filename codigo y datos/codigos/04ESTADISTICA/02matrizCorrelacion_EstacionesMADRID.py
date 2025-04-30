import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

archivo_entrada = "datos/01meteo_horaria_Madrid_largo.csv"

df_meteo = pd.read_csv(archivo_entrada,
                       sep=";", encoding="latin1")

df_meteo["FECHA"] = pd.to_datetime(df_meteo["FECHA"])
# Correlacion horaria
tabla_horaria = df_meteo.pivot(
    index="FECHA", columns="ESTACION", values="PRECIPITACION")
correlacion_horaria = tabla_horaria.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlacion_horaria, cmap="coolwarm",
            vmin=-1, vmax=1, annot=True, fmt=".2f")
plt.title("Matriz de correlación horaria de precipitación")
plt.tight_layout()
plt.show()

# Correlacion diaria
df_meteo["DIA"] = df_meteo["FECHA"].dt.date
df_dia = df_meteo.groupby(["DIA", "ESTACION"])["PRECIPITACION"].sum().unstack()
correlacion_diaria = df_dia.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlacion_diaria, cmap="coolwarm",
            vmin=-1, vmax=1, annot=True, fmt=".2f")
plt.title("Matriz de correlación diaria de precipitación (suma por estación)")
plt.tight_layout()
plt.show()
