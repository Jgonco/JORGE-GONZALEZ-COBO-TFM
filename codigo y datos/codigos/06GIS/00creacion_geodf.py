import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt


archivo_entrada_METEO = "datos/01meteo_horaria_Madrid_largo.csv"
archivo_entrada_Est = "datos/Estaciones_validas.csv"
archivo_entrada_ACCIDENTES = "datos/11accidentes_Madrid.csv"
archivo_salida = "datos/20accidentesyprecipitacion.csv"

# Cargar datos

preciphor_MADRID = pd.read_csv(archivo_entrada_METEO)
estaciones_MADRID = pd.read_csv(archivo_entrada_Est)
accidentes_Madrid = pd.read_csv(archivo_entrada_ACCIDENTES)

# Convertir coordenadas de accidentes a numéricas, manejando errores
accidentes_Madrid['coordenada_x_utm'] = pd.to_numeric(
    accidentes_Madrid['coordenada_x_utm'], errors='coerce')
accidentes_Madrid['coordenada_y_utm'] = pd.to_numeric(
    accidentes_Madrid['coordenada_y_utm'], errors='coerce')


# Convertir estaciones a GeoDataFrame (CRS: EPSG:4258 - ETRS89 lat/lon)
gdf_estaciones = gpd.GeoDataFrame(
    estaciones_MADRID,
    geometry=gpd.points_from_xy(
        estaciones_MADRID.LONGITUD, estaciones_MADRID.LATITUD),
    crs="EPSG:4258"
)

# Convertir accidentes a GeoDataFrame (CRS: EPSG:25830 - ETRS89 UTM Zona 30N)
gdf_accidentes = gpd.GeoDataFrame(
    accidentes_Madrid,
    geometry=gpd.points_from_xy(
        accidentes_Madrid.coordenada_x_utm, accidentes_Madrid.coordenada_y_utm),
    crs="EPSG:25830"
)

# Transformar ambos a un CRS proyectado antes de calcular distancias
gdf_estaciones_proj = gdf_estaciones.to_crs("EPSG:25830")
gdf_accidentes_proj = gdf_accidentes.to_crs("EPSG:25830")

# Encontrar la estación más cercana
gdf_accidentes_proj = gdf_accidentes_proj.sjoin_nearest(
    gdf_estaciones_proj, how="left", distance_col="distancia"
)

# Convertir de vuelta a EPSG:4258 si es necesario
gdf_accidentes = gdf_accidentes_proj.to_crs("EPSG:4258")


gdf_accidentes["FECHA"] = pd.to_datetime(gdf_accidentes["FECHA"])
preciphor_MADRID["FECHA"] = pd.to_datetime(preciphor_MADRID["FECHA"])


gdf_accidentes["ESTACION"] = gdf_accidentes["ESTACION"].astype(str)
preciphor_MADRID["ESTACION"] = preciphor_MADRID["ESTACION"].astype(str)


gdf_accidentes = gdf_accidentes.sort_values("FECHA")
preciphor_MADRID = preciphor_MADRID.sort_values("FECHA")

# Encontrar la precipitación más cercana en el tiempo
df_accyprec = pd.merge_asof(
    gdf_accidentes,
    preciphor_MADRID,
    left_on="FECHA",
    right_on="FECHA",
    by="ESTACION"
)

df_accyprec = df_accyprec[["FECHA", "ESTACION", "PRECIPITACION", "geometry"]]


def convert_geometry(value):
    if isinstance(value, str):
        return wkt.loads(value)
    elif isinstance(value, Point):
        return value
    else:
        return None


# Aplicar conversión a toda la columna
df_accyprec["geometry"] = df_accyprec["geometry"].apply(convert_geometry)

# Extraer LONGITUD y LATITUD
df_accyprec["LONGITUD"] = df_accyprec["geometry"].apply(
    lambda p: p.x if p else None)
df_accyprec["LATITUD"] = df_accyprec["geometry"].apply(
    lambda p: p.y if p else None)
df_accyprec.to_csv(archivo_salida, index=False)
# Convertir el DataFrame a un GeoDataFrame en EPSG:4258
gdf_accyprec = gpd.GeoDataFrame(
    df_accyprec, geometry="geometry", crs="EPSG:4258")
