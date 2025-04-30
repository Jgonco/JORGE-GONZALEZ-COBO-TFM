from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.vectorized import contains

archivo_entrada_METEO = "datos/01meteo_horaria_Madrid_largo.csv"
archivo_entrada_ACCIDENTEs = "datos/11accidentes_Madrid.csv"
archivo_entrada_ESTACIONES = "datos/Estaciones_validas.csv"
archivo_entrada_CONTORNOMAD = "datos/b.shp"

# Cargar archivos
precipitaciones_MADRID_horarias = pd.read_csv(archivo_entrada_METEO, sep=";")
accidentes = pd.read_csv(archivo_entrada_ACCIDENTEs)
estaciones = pd.read_csv(archivo_entrada_ESTACIONES)
contorno_madrid = gpd.read_file(archivo_entrada_CONTORNOMAD)

# Renombrar columnas
accidentes.rename(columns={
    'coordenada_x_utm': 'X',
    'coordenada_y_utm': 'Y'
}, inplace=True)

# Asegurar que las coordenadas sean números
accidentes['X'] = pd.to_numeric(accidentes['X'], errors='coerce')
accidentes['Y'] = pd.to_numeric(accidentes['Y'], errors='coerce')

# Convertir fecha
accidentes['FECHA'] = pd.to_datetime(accidentes['FECHA'], errors='coerce')
accidentes.dropna(subset=['FECHA'], inplace=True)

# Crear GeoDataFrame de accidentes
accidentes['geometry'] = accidentes.apply(
    lambda row: Point(row['X'], row['Y']), axis=1)
accidentes_gdf = gpd.GeoDataFrame(
    accidentes, geometry='geometry', crs='EPSG:25830')

# Precipitación diaria
precipitaciones_MADRID_horarias['FECHA'] = pd.to_datetime(
    precipitaciones_MADRID_horarias['FECHA'])
precipitaciones_MADRID_horarias['AÑO'] = precipitaciones_MADRID_horarias['FECHA'].dt.year
precipitaciones_MADRID_horarias['MES'] = precipitaciones_MADRID_horarias['FECHA'].dt.month
precipitaciones_MADRID_horarias['DÍA'] = precipitaciones_MADRID_horarias['FECHA'].dt.day

precipitaciones_diarias = precipitaciones_MADRID_horarias.groupby(
    ['ESTACION', 'FECHA', 'AÑO', 'MES', 'DÍA']
)['PRECIPITACION'].sum().reset_index()

accidentes_gdf['AÑO'] = accidentes_gdf['FECHA'].dt.year
accidentes_gdf['MES'] = accidentes_gdf['FECHA'].dt.month
accidentes_gdf['DÍA'] = accidentes_gdf['FECHA'].dt.day

# Crear rejilla de interpolación
contorno_madrid = contorno_madrid.to_crs('EPSG:25830')
minx, miny, maxx, maxy = contorno_madrid.total_bounds
grid_x, grid_y = np.mgrid[minx:maxx:200j, miny:maxy:200j]
grid_points_x = grid_x.flatten()
grid_points_y = grid_y.flatten()
madrid_union = contorno_madrid.union_all()
mask_geom = contains(madrid_union, grid_points_x,
                     grid_points_y).reshape(grid_x.shape)

# Crear GeoDataFrame de estaciones
estaciones['geometry'] = estaciones.apply(
    lambda row: Point(row['LONGITUD'], row['LATITUD']), axis=1)
estaciones_gdf = gpd.GeoDataFrame(
    estaciones, geometry='geometry', crs='EPSG:4258').to_crs('EPSG:25830')

# Crear carpeta de salida
os.makedirs("Mapas_diarios", exist_ok=True)

# Parámetros de visualización
min_prec = 1
max_prec = 40

# interpolación IDW


def idw_interpolation(xy, values, grid_xx, grid_yy, power=2.5):
    grid_zz = np.zeros_like(grid_xx)
    for i in range(grid_xx.shape[0]):
        for j in range(grid_xx.shape[1]):
            gx, gy = grid_xx[i, j], grid_yy[i, j]
            dists = np.sqrt((xy[:, 0] - gx)**2 + (xy[:, 1] - gy)**2)
            if np.any(dists == 0):
                grid_zz[i, j] = values[dists == 0][0]
            else:
                weights = 1 / dists**power
                grid_zz[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_zz


# Creación mapas
for (año, mes, dia), grupo in precipitaciones_diarias.groupby(['AÑO', 'MES', 'DÍA']):

    suma_dia = grupo.groupby('ESTACION')['PRECIPITACION'].sum().reset_index()
    suma_dia = suma_dia.merge(
        estaciones[['ESTACION', 'LONGITUD', 'LATITUD']], on='ESTACION', how='left')
    suma_dia = suma_dia.dropna(subset=['LONGITUD', 'LATITUD'])

    puntos = gpd.GeoDataFrame(suma_dia, geometry=gpd.points_from_xy(
        suma_dia['LONGITUD'], suma_dia['LATITUD']), crs='EPSG:4258').to_crs('EPSG:25830')
    puntos_coords = np.array([[point.x, point.y] for point in puntos.geometry])
    valores = suma_dia['PRECIPITACION'].values

    grid_z = idw_interpolation(
        puntos_coords, valores, grid_x, grid_y, power=2.5)
    grid_z_masked = np.where(mask_geom, grid_z, np.nan)

    acc_dia = accidentes_gdf[(accidentes_gdf['AÑO'] == año) &
                             (accidentes_gdf['MES'] == mes) &
                             (accidentes_gdf['DÍA'] == dia)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(f'Precipitación y Accidentes - {año}-{mes:02d}-{dia:02d}')
    im = ax.imshow(grid_z_masked.T, extent=(minx, maxx, miny, maxy), origin='lower',
                   cmap='Blues', alpha=0.8, vmin=min_prec, vmax=max_prec)
    contorno_madrid.boundary.plot(ax=ax, color='black', linewidth=1)

    if len(acc_dia) > 0:
        acc_dia.to_crs('EPSG:25830').plot(
            ax=ax, color='darkred', markersize=10, alpha=0.6)
    else:
        ax.text(0.5, 0.5, 'Sin accidentes registrados', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    for _, row in puntos.iterrows():
        ax.text(row.geometry.x, row.geometry.y, f"{row['PRECIPITACION']:.1f} mm",
                fontsize=8, ha='center', va='bottom', color='black',
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

    ax.text(0.01, 0.99, f'Nº accidentes: {len(acc_dia)}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    plt.colorbar(im, ax=ax, label="Precipitación (mm)")
    plt.tight_layout()
    plt.savefig(f'Mapas_diarios/Mapa_{año}_{mes:02d}_{dia:02d}.png')
    plt.close()
