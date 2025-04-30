import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random
import tensorflow as tf

SEED = 6
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

archivo_entrada = "datos/01meteo_horaria_Madrid_largo.csv"
df_meteo = pd.read_csv(archivo_entrada, sep=";")
df_meteo['FECHA'] = pd.to_datetime(df_meteo['FECHA'], format="%Y-%m-%d %H")
df_meteo['year'] = df_meteo['FECHA'].dt.year
df_meteo['month'] = df_meteo['FECHA'].dt.month
df_meteo['day'] = df_meteo['FECHA'].dt.day

estacion_id = 24

# Filtrar por estación y agregación diaria
df_est = df_meteo[df_meteo['ESTACION'] == estacion_id]
df_diario = df_est.resample('D', on='FECHA').agg({
    'PRECIPITACION': 'sum',
    'TEMPERATURA': 'mean',
    'HUMEDADR': 'mean'
}).reset_index()

df_diario['year'] = df_diario['FECHA'].dt.year
df_diario['month'] = df_diario['FECHA'].dt.month
df_diario['day'] = df_diario['FECHA'].dt.day
# Entrenamiento y test
train_df = df_diario[(df_diario['year'] >= 2019) &
                     (df_diario['year'] <= 2023)].copy()
test_df = df_diario[df_diario['year'] == 2024].copy()

# Modelo Random Forest - Ocurrencia
train_df['lluvia'] = (train_df['PRECIPITACION'] > 0).astype(int)
modelo_ocurrencia_RF = RandomForestClassifier(
    n_estimators=500,
    max_features=4,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 5},
    random_state=SEED
)
modelo_ocurrencia_RF.fit(
    train_df[['day', 'month', 'TEMPERATURA', 'HUMEDADR']], train_df['lluvia'])
# Modelo Random Forest - Intensidad
train_lluvia = train_df[train_df['PRECIPITACION'] > 0].copy()

modelo_intensidad_RF = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    random_state=SEED
)
modelo_intensidad_RF.fit(
    train_lluvia[['day', 'month', 'TEMPERATURA', 'HUMEDADR']],
    train_lluvia['PRECIPITACION']
)

# Predicción
probs = modelo_ocurrencia_RF.predict_proba(
    test_df[['day', 'month', 'TEMPERATURA', 'HUMEDADR']])
prob_lluvia = probs[:, 1] if probs.shape[1] == 2 else np.zeros(len(test_df))

intensidad = modelo_intensidad_RF.predict(
    test_df[['day', 'month', 'TEMPERATURA', 'HUMEDADR']])

test_df['prob_lluvia'] = prob_lluvia
test_df['intensidad_lluvia'] = intensidad

UMBRAL = 0.80
test_df['prediccion_final'] = np.where(prob_lluvia > UMBRAL, intensidad, 0)

# Métricas diarias
mae = mean_absolute_error(
    test_df['PRECIPITACION'], test_df['prediccion_final'])
rmse = mean_squared_error(
    test_df['PRECIPITACION'], test_df['prediccion_final']) ** 0.5
media = test_df['PRECIPITACION'].mean()
desv = test_df['PRECIPITACION'].std()

print(f"Estación: {estacion_id}")
print(f"MAE: {round(mae, 2)} mm")
print(f"RMSE: {round(rmse, 2)} mm")
print(f"Media observada: {round(media, 2)} mm")
print(f"Desviación estándar: {round(desv, 2)} mm")

# Gráfico diario
plt.figure(figsize=(14, 6))
plt.plot(test_df['FECHA'], test_df['PRECIPITACION'],
         label='Observado', linewidth=1.5)
plt.plot(test_df['FECHA'], test_df['prediccion_final'],
         label='Predicho', linestyle='--', linewidth=1.5)
plt.title(f"Estación {estacion_id} - Predicción vs Observado (diario)")
plt.xlabel("Fecha")
plt.ylabel("Precipitación diaria (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Agregación y métricas semanales
test_df['semana'] = test_df['FECHA'].dt.isocalendar().week
df_semanal = test_df.groupby('semana').agg(
    Observado=('PRECIPITACION', 'sum'),
    Predicho=('prediccion_final', 'sum')
).reset_index()

mae_sem = mean_absolute_error(df_semanal['Observado'], df_semanal['Predicho'])
rmse_sem = mean_squared_error(
    df_semanal['Observado'], df_semanal['Predicho']) ** 0.5
media_obs_sem = df_semanal['Observado'].mean()
desv_obs_sem = df_semanal['Observado'].std()

print("Análisis semanal:")
print(f"MAE semanal: {round(mae_sem, 2)} mm")
print(f"RMSE semanal: {round(rmse_sem, 2)} mm")
print(f"Media observada semanal: {round(media_obs_sem, 2)} mm")
print(f"Desviación estándar semanal: {round(desv_obs_sem, 2)} mm")

# Gráfico semanal
plt.figure(figsize=(14, 5))
plt.plot(df_semanal['semana'], df_semanal['Observado'],
         label='Observado', linewidth=2)
plt.plot(df_semanal['semana'], df_semanal['Predicho'],
         label='Predicho', linestyle='--', linewidth=2)
plt.title(f"Estación {estacion_id} - Predicción vs Observado (semanal)")
plt.xlabel("Semana")
plt.ylabel("Precipitación acumulada (mm)")
plt.xticks(ticks=np.arange(1, 54, 2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_export_rfcr = df_semanal[['semana', 'Predicho']].copy()
df_export_rfcr.rename(columns={
    'semana': 'FECHA',
    'Predicho': 'pred_RFCR'
}, inplace=True)

df_export_rfcr.to_csv("datos/predicciones_RFCR.csv", index=False)
