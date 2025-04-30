import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GaussianNoise, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random

SEED = 7

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

# Variable binaria precipitación
df_diario['LLUEVE'] = (df_diario['PRECIPITACION'] > 0).astype(int)

# Normalizar temperatura y humedad
scaler = MinMaxScaler()
df_diario[['TEMPERATURA', 'HUMEDADR']] = scaler.fit_transform(
    df_diario[['TEMPERATURA', 'HUMEDADR']])


# Creación de secuencias
def crear_secuencias(df, look_back=30):
    X, y_bin, y_reg = [], [], []
    for i in range(len(df) - look_back):
        secuencia = df.iloc[i:i+look_back][['PRECIPITACION',
                                            'TEMPERATURA', 'HUMEDADR']].values
        X.append(secuencia)
        y_bin.append(df.iloc[i+look_back]['LLUEVE'])
        y_reg.append(df.iloc[i+look_back]['PRECIPITACION'])
    return np.array(X), np.array(y_bin), np.array(y_reg)


X, y_bin, y_reg = crear_secuencias(df_diario)
fechas = df_diario['FECHA'].iloc[30:].reset_index(drop=True)
train_mask = fechas < pd.to_datetime("2024-01-01")
X_train, X_test = X[train_mask], X[~train_mask]
y_bin_train, y_bin_test = y_bin[train_mask], y_bin[~train_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[~train_mask]

# Modelo LSTM-Ocurrencia
clf_model = Sequential([
    Input(shape=(30, 3)),
    GaussianNoise(0.001),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1),
    Activation('sigmoid')
])

clf_model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

clf_model.fit(X_train, y_bin_train,
              validation_data=(X_test, y_bin_test),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop],
              verbose=0)

# Modelo LSTM-Intensidad
reg_model = Sequential([
    Input(shape=(30, 3)),
    GaussianNoise(0.001),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1),
    Activation('relu')
])

reg_model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

reg_model.fit(X_train, y_reg_train,
              validation_data=(X_test, y_reg_test),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop],
              verbose=0)

# Predicción combinada
y_pred_bin = clf_model.predict(X_test).flatten()
UMBRAL = 0.35
y_pred_llueve = (y_pred_bin > UMBRAL).astype(int)

y_pred_reg = reg_model.predict(X_test).flatten()
y_final_pred = y_pred_reg * y_pred_llueve
y_real = y_reg_test
y_final_pred = np.maximum(y_final_pred, 0)

# Métricas diarias
rmse = np.sqrt(mean_squared_error(y_real, y_final_pred))
mae = mean_absolute_error(y_real, y_final_pred)

print(f"MAE combinado: {mae:.2f} mm")
print(f"RMSE combinado: {rmse:.2f} mm")
media_real = np.mean(y_real)
std_real = np.std(y_real)

print(f"Media real de precipitación: {media_real:.2f} mm")
print(f"Desviación estándar real: {std_real:.2f} mm")

# Gráfica diaria
plt.figure(figsize=(14, 6))
plt.plot(y_real, label='Real')
plt.plot(y_final_pred, label='Predicho')
plt.title("Predicción de precipitación diaria combinada (Estación 24)")
plt.xlabel("Días de 2024")
plt.ylabel("Precipitación (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Crear DataFrame con fechas, predicciones y observaciones
df_resultados = pd.DataFrame({
    'fecha': fechas[~train_mask].reset_index(drop=True),
    'observado': y_real,
    'predicho': y_final_pred
})

# Agregar por semana
df_resultados['semana'] = df_resultados['fecha'].dt.isocalendar().week
df_semanal = df_resultados.groupby('semana').agg({
    'observado': 'sum',
    'predicho': 'sum'
}).reset_index()

# Métricas semanales
rmse_sem = np.sqrt(mean_squared_error(
    df_semanal['observado'], df_semanal['predicho']))
mae_sem = mean_absolute_error(df_semanal['observado'], df_semanal['predicho'])
media_obs_sem = df_semanal['observado'].mean()
desv_obs_sem = df_semanal['observado'].std()

print("Evaluación semanal:")
print(f"MAE semanal: {mae_sem:.2f} mm")
print(f"RMSE semanal: {rmse_sem:.2f} mm")
print(f"Media observada semanal: {media_obs_sem:.2f} mm")
print(f"Desviación estándar semanal: {desv_obs_sem:.2f} mm")

# Gráfico semanal
plt.figure(figsize=(14, 5))
plt.plot(df_semanal['semana'], df_semanal['observado'],
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


df_export_lstm_dual = df_semanal[['semana', 'predicho']].copy()
df_export_lstm_dual.rename(columns={
    'semana': 'FECHA',
    'predicho': 'pred_LSTMDUAL'
}, inplace=True)

df_export_lstm_dual.to_csv("predicciones_LSTMDUAL.csv", index=False)
