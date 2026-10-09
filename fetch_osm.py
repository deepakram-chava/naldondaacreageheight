import requests
import json
import os

overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
[out:json][timeout:60];
area["name:en"="Nalgonda"]["admin_level"="5"]->.searchArea;
(
  way["landuse"="farmland"](area.searchArea);
  relation["landuse"="farmland"](area.searchArea);
);
out geom;
"""
print("Querying Overpass API...")
try:
    response = requests.post(overpass_url, data={'data': overpass_query})
    data = response.json()
    features = []
    for element in data['elements']:
        if element['type'] == 'way':
            coords = [[node['lon'], node['lat']] for node in element['geometry']]
            if coords[0] != coords[-1]:
                coords.append(coords[0]) # close polygon
            geom = {"type": "Polygon", "coordinates": [coords]}
            features.append({"type": "Feature", "geometry": geom, "properties": {"crop": "Farmland (OSM)"}})
    
    geojson = {"type": "FeatureCollection", "features": features}
    
    os.makedirs('data/nalgonda/crop_layers', exist_ok=True)
    with open('data/nalgonda/crop_layers/osm_farmland.geojson', 'w') as f:
        json.dump(geojson, f)
    print(f"Downloaded {len(features)} farmland polygons from OSM.")
except Exception as e:
    print(e)
