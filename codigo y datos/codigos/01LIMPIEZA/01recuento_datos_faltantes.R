install.packages("readr")
install.packages("dplyr")
install.packages("naniar")

library(readr)
library(dplyr)
library(naniar)

precipitacionesH <- read_delim("datos/00brutosprecip_horaria_MADRID.csv",
  delim = ";", escape_double = FALSE, trim_ws = TRUE
)

# Contar valores faltantes (NA) por estación
faltantes_por_estacion <- precipitacionesH %>%
  group_by(ESTACION) %>%
  summarise(across(starts_with("H"), ~ sum(is.na(.))))



print(faltantes_por_estacion)
