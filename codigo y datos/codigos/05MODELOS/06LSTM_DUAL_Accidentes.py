import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GaussianNoise, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

archivo_entrada = "datos/21accidentesporestacion.csv"

df_accidentes = pd.read_csv(archivo_entrada)
df_accidentes['FECHA'] = pd.to_datetime(df_accidentes['FECHA'])
estacion_id = 24

df_accidentes = df_accidentes[df_accidentes['ESTACION'] == estacion_id]

df_accidentes = df_accidentes.set_index('FECHA').sort_index()

# Variable binaria accidente
df_accidentes['OCURRE'] = (df_accidentes['ACCIDENTES'] > 0).astype(int)

# Normalizar precipitación
scaler = MinMaxScaler()
df_accidentes[['PRECIPITACION']] = scaler.fit_transform(
    df_accidentes[['PRECIPITACION']])


# Creación de secuencias
def crear_secuencias(df, look_back=48):
    X, y_bin, y_reg = [], [], []
    for i in range(len(df) - look_back):
        secuencia = df.iloc[i:i +
                            look_back][['ACCIDENTES', 'PRECIPITACION']].values
        X.append(secuencia)
        y_bin.append(df.iloc[i+look_back]['OCURRE'])
        y_reg.append(df.iloc[i+look_back]['ACCIDENTES'])
    return np.array(X), np.array(y_bin), np.array(y_reg)


X, y_bin, y_reg = crear_secuencias(df_accidentes)
fechas = df_accidentes.index[48:]
train_mask = fechas < pd.to_datetime("2024-01-01")

X_train, X_test = X[train_mask], X[~train_mask]
y_bin_train, y_bin_test = y_bin[train_mask], y_bin[~train_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[~train_mask]
fechas_test = fechas[~train_mask]

# Modelo LSTM-Ocurrencia
clf_model = Sequential([
    Input(shape=(48, 2)),
    GaussianNoise(0.001),
    LSTM(128),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

clf_model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

clf_model.fit(X_train, y_bin_train,
              validation_data=(X_test, y_bin_test),
              epochs=50,
              batch_size=32,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

# Modelo LSTM-Intensidad
reg_model = Sequential([
    Input(shape=(48, 2)),
    GaussianNoise(0.001),
    LSTM(128),
    Dropout(0.2),
    Dense(1),
    Activation('relu')
])

reg_model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

reg_model.fit(X_train, y_reg_train,
              validation_data=(X_test, y_reg_test),
              epochs=50,
              batch_size=32,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

# Predicción
y_pred_bin = clf_model.predict(X_test).flatten()
UMBRAL = 0.25
y_pred_ocurre = (y_pred_bin > UMBRAL).astype(int)

y_pred_reg = reg_model.predict(X_test).flatten()
y_pred_final = y_pred_reg * y_pred_ocurre

# Crear DataFrame con resultados
df_resultados = pd.DataFrame({
    'fecha': fechas_test,
    'real': y_reg_test,
    'predicho': y_pred_final
})

# Agregación diaria
df_resultados.set_index('fecha', inplace=True)
df_diario = df_resultados.resample('D').sum().reset_index()

# Métricas diarias
mae_diario = mean_absolute_error(df_diario['real'], df_diario['predicho'])
rmse_diario = np.sqrt(mean_squared_error(
    df_diario['real'], df_diario['predicho']))

print(f"MAE diario: {mae_diario:.2f}")
print(f"RMSE diario: {rmse_diario:.2f}")

# Gráfico diario
plt.figure(figsize=(14, 6))
plt.plot(df_diario['fecha'], df_diario['real'], label='Real', linewidth=1.5)
plt.plot(df_diario['fecha'], df_diario['predicho'],
         label='Predicho', linestyle='--', linewidth=1.5)
plt.title("Predicción diaria de accidentes - Estación 24")
plt.xlabel("Fecha")
plt.ylabel("Accidentes diarios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Agregación semanal
df_resultados = df_resultados.reset_index()
df_resultados['semana'] = df_resultados['fecha'].dt.isocalendar().week
df_semanal = df_resultados.groupby('semana').agg(
    {'real': 'sum', 'predicho': 'sum'}).reset_index()

# Métricas semanales
mae_semanal = mean_absolute_error(df_semanal['real'], df_semanal['predicho'])
rmse_semanal = np.sqrt(mean_squared_error(
    df_semanal['real'], df_semanal['predicho']))
media_semanal_real = df_semanal['real'].mean()
std_semanal_real = df_semanal['real'].std()

print("Evaluación semanal:")
print(f"MAE semanal: {mae_semanal:.2f}")
print(f"RMSE semanal: {rmse_semanal:.2f}")
print(f"Media semanal real: {media_semanal_real:.2f}")
print(f"Desviación estándar semanal: {std_semanal_real:.2f}")

# Gráfico semanal
plt.figure(figsize=(14, 5))
plt.plot(df_semanal['semana'], df_semanal['real'], label='Real', linewidth=2)
plt.plot(df_semanal['semana'], df_semanal['predicho'],
         label='Predicho', linestyle='--', linewidth=2)
plt.title("Predicción semanal de accidentes - Estación 24")
plt.xlabel("Semana del año")
plt.ylabel("Accidentes acumulados")
plt.xticks(np.arange(1, 54, 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
