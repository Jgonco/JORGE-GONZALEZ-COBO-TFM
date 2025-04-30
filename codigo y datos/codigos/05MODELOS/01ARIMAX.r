install.packages(c("forecast", "ggplot2", "Metrics", "lubridate", "dplyr", "gridExtra", "tseries"))

library(forecast)
library(ggplot2)
library(Metrics)
library(lubridate)
library(dplyr)
library(gridExtra)
library(tseries)


df_precip <- X02precip_horaria_Madrid_largo_estaciones
df_temp <- X02temperatura_horaria_Madrid_largo_estaciones
df_humedad <- X02humedad_horaria_Madrid_largo_estaciones

df_precip$FECHA <- as.POSIXct(df_precip$FECHA)
df_temp$FECHA <- as.POSIXct(df_temp$FECHA)
df_humedad$FECHA <- as.POSIXct(df_humedad$FECHA)

df_meteo <- df_precip %>%
  select(FECHA, S24_precip = S24) %>%
  left_join(df_temp %>% select(FECHA, S24_temp = S24), by = "FECHA") %>%
  left_join(df_humedad %>% select(FECHA, S24_hum = S24), by = "FECHA") %>%
  filter(!is.na(S24_precip), !is.na(S24_temp), !is.na(S24_hum)) %>%
  arrange(FECHA)

# Descomposición STL
ts_S24 <- ts(df_meteo$S24_precip, frequency = 365)
decomp <- stl(ts_S24, s.window = "periodic")

df_decomp <- data.frame(
  Fecha = as.Date(df_meteo$FECHA),
  Observado = as.numeric(ts_S24),
  Tendencia = decomp$time.series[, "trend"],
  Estacionalidad = decomp$time.series[, "seasonal"],
  Residuo = decomp$time.series[, "remainder"]
)

# Límites para gráficos
y_lim <- range(df_decomp$Observado, df_decomp$Tendencia,
  df_decomp$Estacionalidad, df_decomp$Residuo,
  na.rm = TRUE
)

# Gráficos de descomposición
g_stl1 <- ggplot(df_decomp, aes(x = Fecha, y = Observado)) +
  geom_line(color = "black") +
  ggtitle("Observado") +
  ylim(y_lim) +
  theme_minimal()

g_stl2 <- ggplot(df_decomp, aes(x = Fecha, y = Tendencia)) +
  geom_line(color = "blue") +
  ggtitle("Tendencia") +
  ylim(y_lim) +
  theme_minimal()

g_stl3 <- ggplot(df_decomp, aes(x = Fecha, y = Estacionalidad)) +
  geom_line(color = "orange") +
  ggtitle("Estacionalidad") +
  ylim(y_lim) +
  theme_minimal()

g_stl4 <- ggplot(df_decomp, aes(x = Fecha, y = Residuo)) +
  geom_line(color = "red") +
  ggtitle("Residuo") +
  ylim(y_lim) +
  theme_minimal()

# Preparación de datos para ARIMAX
ts_all <- ts(df_meteo$S24_precip, frequency = 365)
xreg_all <- as.matrix(df_meteo[, c("S24_temp", "S24_hum")])

train_idx <- df_meteo$FECHA < as.POSIXct("2024-01-01")
test_idx <- df_meteo$FECHA >= as.POSIXct("2024-01-01")

train_ts <- ts(df_meteo$S24_precip[train_idx], frequency = 365)
test_ts <- ts(df_meteo$S24_precip[test_idx], frequency = 365)

xreg_train <- xreg_all[train_idx, ]
xreg_test <- xreg_all[test_idx, ]

# Test estacionariedad
cat("Resultado test Dickey-Fuller:\n")
print(adf.test(train_ts))

# Aplicación del modelo ARIMAX
modelo_arimax <- auto.arima(train_ts, xreg = xreg_train, stationary = TRUE)
print(modelo_arimax)

pred <- forecast(modelo_arimax, xreg = xreg_test)

# Métricas diarias
mae_diaria <- mae(test_ts, pred$mean)
rmse_diaria <- rmse(test_ts, pred$mean)

cat("\nEvaluación diaria (ARIMAX):\n")
cat("MAE:", round(mae_diaria, 2), "mm\n")
cat("RMSE:", round(rmse_diaria, 2), "mm\n")

# Métricas semanales
df_eval <- data.frame(
  fecha = df_meteo$FECHA[test_idx],
  real = as.numeric(test_ts),
  pred = as.numeric(pred$mean)
)

df_eval$semana <- floor_date(df_eval$fecha, "week")

df_semanal <- df_eval %>%
  group_by(semana) %>%
  summarise(real = sum(real), pred = sum(pred), .groups = "drop")

mae_semanal <- mae(df_semanal$real, df_semanal$pred)
rmse_semanal <- rmse(df_semanal$real, df_semanal$pred)

cat("\nEvaluación semanal (ARIMAX):\n")
cat("MAE semanal:", round(mae_semanal, 2), "mm\n")
cat("RMSE semanal:", round(rmse_semanal, 2), "mm\n")

# Gráficos diario y semanal
options(repr.plot.width = 12, repr.plot.height = 24)

g1 <- ggplot(df_eval, aes(x = fecha)) +
  geom_line(aes(y = real, color = "Real")) +
  geom_line(aes(y = pred, color = "Predicción"), linetype = "dashed") +
  labs(title = "Predicción diaria - ARIMAX", y = "Precipitación (mm)", color = "") +
  theme_minimal()

g2 <- ggplot(df_semanal, aes(x = semana)) +
  geom_line(aes(y = real, color = "Real")) +
  geom_line(aes(y = pred, color = "Predicción"), linetype = "dashed") +
  labs(title = "Predicción semanal - ARIMAX", y = "Precipitación acumulada (mm)", color = "") +
  theme_minimal()

# Mostrar gráficas
grid.arrange(g_stl1, g_stl2, g_stl3, g_stl4, ncol = 1)
grid.arrange(g1, g2, ncol = 1)


# Crear DataFrame de salida
df_export <- df_semanal %>%
  rename(FECHA = semana, REAL = real) %>%
  mutate(Pred_ARIMAX = pred) %>%
  select(FECHA, REAL, pred_ARIMAX)

# Guardar CSV
write.csv(df_export, "predicciones_ARIMAX.csv", row.names = FALSE)
