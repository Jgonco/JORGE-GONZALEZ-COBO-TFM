import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


archivo_arimax = "datos/predicciones_ARIMAX.csv"
archivo_rf = "datos/predicciones_RF.csv"
archivo_rfcr = "datos/predicciones_RFCR.csv"
archivo_lstm = "datos/predicciones_LSTM.csv"
archivo_lstmdual = "datos/predicciones_LSTMDUAL.csv"

df_arimax = pd.read_csv(archivo_arimax)
df_rf = pd.read_csv(archivo_rf)
df_rfcr = pd.read_csv(archivo_rfcr)
df_lstm = pd.read_csv(archivo_lstm)
df_lstmdual = pd.read_csv(archivo_lstmdual)

# Unir datos
df_final = df_arimax.copy()
df_final = df_final.merge(df_rf, on="FECHA", how="left")
df_final = df_final.merge(df_rfcr, on="FECHA", how="left")
df_final = df_final.merge(df_lstm, on="FECHA", how="left")
df_final = df_final.merge(df_lstmdual, on="FECHA", how="left")

# Gráfico comparativo
plt.figure(figsize=(14, 6))

plt.plot(df_final['FECHA'], df_final['REAL'], label='Real', linewidth=3)
plt.plot(df_final['FECHA'], df_final['pred_ARIMAX'],
         label='ARIMAX', linestyle='--', linewidth=1)
plt.plot(df_final['FECHA'], df_final['pred_RF'],
         label='RF', linestyle='--', linewidth=1)
plt.plot(df_final['FECHA'], df_final['pred_RFCR'],
         label='RFCR', linestyle='--', linewidth=1)
plt.plot(df_final['FECHA'], df_final['pred_LSTM'],
         label='LSTM', linestyle='--', linewidth=1)
plt.plot(df_final['FECHA'], df_final['pred_LSTMDUAL'],
         label='LSTM Dual', linewidth=1.5)

plt.title("Predicción semanal de precipitación - Comparativa de modelos")
plt.xlabel("Semana")
plt.ylabel("Precipitación acumulada (mm)")
plt.xticks(ticks=np.arange(1, 54, 2))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
