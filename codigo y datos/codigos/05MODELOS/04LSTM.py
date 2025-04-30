import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GaussianNoise, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import random

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

archivo_entrada = "01meteo_horaria_Madrid_largo.csv"

df_meteo = pd.read_csv(archivo_entrada, sep=";")
df_meteo['FECHA'] = pd.to_datetime(df_meteo['FECHA'], format='%Y-%m-%d %H')
estacion_id = 24

# Filtrar por estación y agregación diaria
df_meteo = df_meteo[df_meteo['ESTACION'] == estacion_id]
df_diario = df_meteo.set_index('FECHA').resample('D').agg({
    'PRECIPITACION': 'sum',
    'TEMPERATURA': 'mean',
    'HUMEDADR': 'mean'
}).reset_index()

scaler = MinMaxScaler()
df_diario[['TEMPERATURA', 'HUMEDADR']] = scaler.fit_transform(
    df_diario[['TEMPERATURA', 'HUMEDADR']]
)


# Funcion para agregar ruido gaussiano
def agregar_ruido_gaussiano(serie, std=0.001):
    ruido = np.random.normal(0, std, size=serie.shape)
    resultado = serie.copy()
    resultado[serie == 0] += ruido[serie == 0]
    resultado[resultado < 0] = 0
    return resultado


df_diario['PRECIPITACION_NOISY'] = agregar_ruido_gaussiano(
    df_diario['PRECIPITACION'])


# Creación de secuencias
def crear_secuencias(df, ventana=30):
    X, y = [], []
    for i in range(len(df) - ventana):
        X.append(df.iloc[i:i+ventana][['PRECIPITACION_NOISY',
                 'TEMPERATURA', 'HUMEDADR']].values)
        y.append(df.iloc[i+ventana]['PRECIPITACION'])
    return np.array(X), np.array(y)


X, y = crear_secuencias(df_diario)
index_fechas = df_diario['FECHA'].iloc[30:].reset_index(drop=True)
train_mask = index_fechas < pd.to_datetime("2024-01-01")
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Modelo LSTM
model = Sequential([
    Input(shape=(30, 3)),
    GaussianNoise(0.001),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1),
    Activation('relu')
])

model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop],
          verbose=0)

# Predicción y aplicación de umbral
y_pred = model.predict(X_test).flatten()
UMBRAL = 0.85
y_pred = np.where(y_pred > UMBRAL, y_pred, 0)

# Métricas diarias
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
media_real = np.mean(y_test)
std_real = np.std(y_test)

print(f'MAE en test (2024): {mae:.2f} mm')
print(f'RMSE en test (2024): {rmse:.2f} mm')
print(f"Media real de precipitación: {media_real:.2f} mm")
print(f"Desviación estándar real: {std_real:.2f} mm")

# Gráfico diario
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Predicho', linestyle='--')
plt.title('Predicción de precipitación diaria - Estación 24')
plt.xlabel("Fecha")
plt.ylabel("Precipitación (mm)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


df_resultados = pd.DataFrame({
    'fecha': index_fechas[~train_mask].reset_index(drop=True),
    'real': y_test,
    'predicho': y_pred
})
# Agregación y métricas semanales
df_resultados['semana'] = df_resultados['fecha'].dt.isocalendar().week

df_semanal = df_resultados.groupby('semana').agg({
    'real': 'sum',
    'predicho': 'sum'
}).reset_index()

rmse_sem = np.sqrt(mean_squared_error(
    df_semanal['real'], df_semanal['predicho']))
mae_sem = mean_absolute_error(df_semanal['real'], df_semanal['predicho'])
media_sem = df_semanal['real'].mean()
std_sem = df_semanal['real'].std()

print("Evaluación semanal:")
print(f"MAE semanal: {mae_sem:.2f} mm")
print(f"RMSE semanal: {rmse_sem:.2f} mm")
print(f"Media observada semanal: {media_sem:.2f} mm")
print(f"Desviación estándar semanal: {std_sem:.2f} mm")

# Gráfico semanal
plt.figure(figsize=(14, 5))
plt.plot(df_semanal['semana'], df_semanal['real'],
         label='Observado', linewidth=2)
plt.plot(df_semanal['semana'], df_semanal['predicho'],
         label='Predicho', linestyle='--', linewidth=2)
plt.title("Predicción vs Observado (precipitación semanal) - Estación 24")
plt.xlabel("Semana")
plt.ylabel("Precipitación acumulada (mm)")
plt.xticks(ticks=np.arange(1, 54, 2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_export_lstm = df_semanal[['semana', 'predicho']].copy()
df_export_lstm.rename(columns={
    'semana': 'FECHA',
    'predicho': 'pred_LSTM'
}, inplace=True)

df_export_lstm.to_csv("datos/predicciones_LSTM.csv", index=False)
