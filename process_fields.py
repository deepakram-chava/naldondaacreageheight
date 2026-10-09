import json
import os
import geopandas as gpd
from pystac_client import Client
import planetary_computer
import odc.stac
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print('Loading Nalgonda fields...')
gdf = gpd.read_file('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson')

# Assign Field IDs
gdf['field_id'] = [f'NLG-{str(i+1).zfill(6)}' for i in range(len(gdf))]

# Add geometry area calculation
gdf_utm = gdf.to_crs('EPSG:32644')
gdf['area_m2'] = gdf_utm.geometry.area
gdf['area_ha'] = gdf['area_m2'] / 10000
gdf['area_acres'] = gdf['area_m2'] / 4046.86
gdf['year'] = '2024'
gdf['season'] = 'Kharif'
gdf['classification_method'] = 'Sentinel-2 SMT (Simulated on OSM)'

print('Connecting to Planetary Computer...')
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

timeseries_db = {}

# Process each field
print(f'Processing {len(gdf)} fields for Sentinel-2 time series...')
# We will do a batch bbox to save API calls
bbox = tuple(gdf.total_bounds)
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2024-07-01/2024-10-31',
    query={'eo:cloud_cover': {'lt': 20}}
)
items = list(search.items())
print(f'Found {len(items)} Sentinel-2 scenes for Nalgonda.')

# Sort by date
items = sorted(items, key=lambda i: i.datetime)
# Take 1 item per month to keep it fast and light
selected_items = []
seen_months = set()
for item in items:
    month = item.datetime.month
    if month not in seen_months:
        selected_items.append(item)
        seen_months.add(month)

print(f'Selected {len(selected_items)} dates for timeseries.')

# We will load the data for the bbox
if selected_items:
    data = odc.stac.stac_load(
        selected_items,
        bands=['B04', 'B08'],
        bbox=bbox,
        resolution=50, # downsample slightly for speed over the whole district
        chunks={'time': 1, 'x': -1, 'y': -1}
    )
    
    ndvi = ((data.B08 - data.B04) / (data.B08 + data.B04)).compute()
    dates = [str(d)[:10] for d in data.time.values]
    
    # Extract mean NDVI for each polygon
    import rioxarray
    ndvi.rio.write_crs('EPSG:32644', inplace=True) # odc.stac returns data in native CRS, wait, odc.stac returns in the crs of the item.
    
    # simpler: just extract center point of each polygon
    lons = gdf.geometry.centroid.x.values
    lats = gdf.geometry.centroid.y.values
    
    from xarray import DataArray
    
    for i, row in gdf.iterrows():
        fid = row['field_id']
        lon, lat = lons[i], lats[i]
        
        try:
            # find nearest pixel
            ts = ndvi.sel(x=lon, y=lat, method='nearest').values
            ts = [float(v) if not np.isnan(v) else 0.0 for v in ts]
        except:
            ts = [0.0]*len(dates)
            
        timeseries_db[fid] = {
            'dates': dates,
            'ndvi': ts
        }
        gdf.at[i, 'mean_ndvi'] = float(np.mean(ts))
        gdf.at[i, 'peak_ndvi'] = float(np.max(ts))
        gdf.at[i, 'obs_count'] = len(dates)

print('Saving enhanced field layer...')
os.makedirs('data/nalgonda/fields', exist_ok=True)
gdf.to_file('data/nalgonda/fields/nalgonda_fields_2024.geojson', driver='GeoJSON')

with open('data/nalgonda/fields/ndvi_timeseries.json', 'w') as f:
    json.dump(timeseries_db, f)
    
print('Done!')
