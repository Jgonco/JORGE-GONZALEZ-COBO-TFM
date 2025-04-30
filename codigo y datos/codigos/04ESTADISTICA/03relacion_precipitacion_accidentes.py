import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, normaltest
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Carga de datos
archivo_entrada = "datos/21accidentesporestacion.csv"
df_accidentes = pd.read_csv(archivo_entrada)
df_accidentes['FECHA'] = pd.to_datetime(df_accidentes['FECHA'])
df_accidentes.set_index('FECHA', inplace=True)

# Agregación diaria
df_diario = df_accidentes.resample('D').agg({
    'ACCIDENTES': 'sum',
    'PRECIPITACION': 'sum'
}).dropna()

df_diario['LLUEVE'] = (df_diario['PRECIPITACION'] > 0).astype(int)

# Gráfico de distribución  de accidentes sin lluvia y con lluvia
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_diario, x='LLUEVE', y='ACCIDENTES')
plt.xticks([0, 1], ['Sin lluvia', 'Con lluvia'])
plt.title("Distribución diaria de accidentes según presencia de precipitación")
plt.ylabel("Nº de accidentes diarios")
plt.xlabel("Precipitación")
plt.grid(True)
plt.tight_layout()
plt.show()

# Prueba de normalidad D’Agostino
acc_sin_lluvia = df_diario[df_diario['LLUEVE'] == 0]['ACCIDENTES']
acc_con_lluvia = df_diario[df_diario['LLUEVE'] == 1]['ACCIDENTES']

stat_sin, p_sin = normaltest(acc_sin_lluvia)
stat_con, p_con = normaltest(acc_con_lluvia)

print(
    f'Sin lluvia - D’Agostino: estadístico={stat_sin:.3f}, p-valor={p_sin:.3f}')
print(
    f'Con lluvia - D’Agostino: estadístico={stat_con:.3f}, p-valor={p_con:.3f}')

# Prueba de U de Mann-Whitney
stat, p = mannwhitneyu(acc_sin_lluvia, acc_con_lluvia, alternative='two-sided')

print(f"Prueba de Mann-Whitney U: {stat:.2f}")
print(f"Valor p: {p:.4f}")
if p < 0.05:
    print("Hay diferencias significativas en el número diario de accidentes entre días con y sin lluvia.")
else:
    print("No hay diferencias significativas.")

# Regresión Binomial Negativa
modelo1 = smf.glm(
    formula='ACCIDENTES ~ PRECIPITACION',
    data=df_diario,
    family=sm.families.NegativeBinomial()
).fit()

print("Modelo Binomial Negativo: ACCIDENTES diarios ~ PRECIPITACION diaria")
print(modelo1.summary())
