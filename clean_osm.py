import json
import os
import geopandas as gpd

gdf = gpd.read_file('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson')

def clean_crop(c):
    c_lower = str(c).lower()
    if 'rice' in c_lower or 'paddy' in c_lower:
        return 'Paddy'
    elif 'cotton' in c_lower:
        return 'Cotton'
    elif 'grass' in c_lower or 'orchard' in c_lower or 'wheat' in c_lower:
        return 'Other Crops'
    else:
        return 'Other Crops'

gdf['crop'] = gdf['crop'].apply(clean_crop)

# Re-save
gdf.to_file('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson', driver='GeoJSON')
print("Cleaned classes:")
print(gdf['crop'].value_counts())
