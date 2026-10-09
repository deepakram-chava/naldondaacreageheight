import json
import os
import geopandas as gpd
from pystac_client import Client
import planetary_computer
import odc.stac
import numpy as np

gdf = gpd.read_file('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson')

# Assign Field IDs
gdf['field_id'] = [f'NLG-{str(i+1).zfill(6)}' for i in range(len(gdf))]
gdf_utm = gdf.to_crs('EPSG:32644')
gdf['area_m2'] = gdf_utm.geometry.area
gdf['area_ha'] = gdf['area_m2'] / 10000
gdf['area_acres'] = gdf['area_m2'] / 4046.86
gdf['year'] = '2024'
gdf['season'] = 'Kharif'
gdf['classification_method'] = 'Sentinel-2 SMT (Simulated on OSM)'

catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

bbox = tuple(gdf.total_bounds)
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2024-01-01/2024-10-31',
    query={'eo:cloud_cover': {'lt': 80}}
)
items = list(search.items())
items = sorted(items, key=lambda i: i.datetime)
selected_items = []
seen_months = set()
for item in items:
    month = item.datetime.month
    if month not in seen_months:
        selected_items.append(item)
        seen_months.add(month)

print(f'Selected {len(selected_items)} dates for timeseries.')

timeseries_db = {}
if selected_items:
    data = odc.stac.stac_load(
        selected_items,
        bands=['B04', 'B08'],
        bbox=bbox,
        resolution=100,
        chunks={'time': 1, 'x': -1, 'y': -1}
    )
    
    ndvi = ((data.B08 - data.B04) / (data.B08 + data.B04)).compute()
    dates = [str(d)[:10] for d in data.time.values]
    
    # Simple extraction
    # odc.stac returns data in Web Mercator or a specific EPSG. Let's reproject gdf.
    crs = data.odc.crs
    gdf_proj = gdf.to_crs(crs)
    
    lons = gdf_proj.geometry.centroid.x.values
    lats = gdf_proj.geometry.centroid.y.values
    
    for i, row in gdf.iterrows():
        fid = row['field_id']
        lon, lat = lons[i], lats[i]
        
        try:
            ts = ndvi.sel(x=lon, y=lat, method='nearest').values
            ts = [float(v) if not np.isnan(v) else 0.0 for v in ts]
        except Exception as e:
            ts = [0.0]*len(dates)
            
        timeseries_db[fid] = {
            'dates': dates,
            'ndvi': ts
        }
        gdf.at[i, 'mean_ndvi'] = float(np.mean(ts))
        gdf.at[i, 'peak_ndvi'] = float(np.max(ts))
        gdf.at[i, 'obs_count'] = len(dates)

os.makedirs('data/nalgonda/fields', exist_ok=True)
gdf.to_file('data/nalgonda/fields/nalgonda_fields_2024.geojson', driver='GeoJSON')
with open('data/nalgonda/fields/ndvi_timeseries.json', 'w') as f:
    json.dump(timeseries_db, f)

print('Done!')
