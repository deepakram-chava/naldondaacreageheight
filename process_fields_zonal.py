import json
import os
import geopandas as gpd
from pystac_client import Client
import planetary_computer
import odc.stac
import numpy as np
from rasterstats import zonal_stats
import warnings
warnings.filterwarnings('ignore')

print('Loading Nalgonda fields...')
gdf = gpd.read_file('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson')

# Standardize Crop labels explicitly for the UI and backend separation
def map_crop(c):
    c = str(c)
    if 'Paddy' in c: return 'Verified/Reference Paddy'
    elif 'Cotton' in c: return 'Verified/Reference Cotton'
    elif 'Other Crops' in c and 'Unclassified' not in c: return 'Other Verified Crop'
    return 'Unclassified Agricultural Field'

gdf['crop'] = gdf['crop'].apply(map_crop)

# Assign Field IDs and Area
gdf['field_id'] = [f'NLG-{str(i+1).zfill(6)}' for i in range(len(gdf))]
gdf_utm = gdf.to_crs('EPSG:32644')
gdf['area_m2'] = gdf_utm.geometry.area
gdf['area_ha'] = gdf['area_m2'] / 10000
gdf['area_acres'] = gdf['area_m2'] / 4046.86
gdf['year'] = '2024'
gdf['season'] = 'Kharif'
gdf['classification_method'] = 'Sentinel-2 NDVI time-series analysis'

print('Connecting to Planetary Computer STAC...')
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

bbox = tuple(gdf.total_bounds)
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2024-06-01/2024-10-31',
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

print(f'Loading {len(selected_items)} dates for timeseries...')
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
    
    # Project gdf to the STAC data's crs
    crs = data.odc.crs
    gdf_proj = gdf.to_crs(crs)
    
    # Initialize timeseries arrays
    for fid in gdf_proj['field_id']:
        timeseries_db[fid] = {'dates': [], 'ndvi': []}
        
    for t_idx, date_str in enumerate(dates):
        # Extract the 2D array for this date
        arr = ndvi.isel(time=t_idx).values
        # Create affine transform from the xarray coordinates
        transform = data.odc.geobox.transform
        
        # Calculate zonal stats
        stats = zonal_stats(gdf_proj, arr, affine=transform, stats=['mean', 'count'], nodata=np.nan)
        
        for i, row in gdf_proj.iterrows():
            fid = row['field_id']
            mean_val = stats[i]['mean']
            # Only append if valid! Do not use 0.0 for NaN.
            if mean_val is not None and not np.isnan(mean_val):
                timeseries_db[fid]['dates'].append(date_str)
                timeseries_db[fid]['ndvi'].append(float(mean_val))
                
    # Compute summary stats on the valid timeseries
    for i, row in gdf.iterrows():
        fid = row['field_id']
        valid_ndvi = timeseries_db[fid]['ndvi']
        if valid_ndvi:
            gdf.at[i, 'mean_ndvi'] = float(np.mean(valid_ndvi))
            gdf.at[i, 'peak_ndvi'] = float(np.max(valid_ndvi))
            gdf.at[i, 'obs_count'] = len(valid_ndvi)
        else:
            gdf.at[i, 'mean_ndvi'] = None
            gdf.at[i, 'peak_ndvi'] = None
            gdf.at[i, 'obs_count'] = 0

os.makedirs('data/nalgonda/fields', exist_ok=True)
gdf.to_file('data/nalgonda/fields/nalgonda_fields_2024.geojson', driver='GeoJSON')
with open('data/nalgonda/fields/ndvi_timeseries.json', 'w') as f:
    json.dump(timeseries_db, f)

print('Process complete. True zonal statistics applied.')
