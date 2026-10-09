import urllib.request
import urllib.parse
import json
import os

overpass_url = 'https://overpass-api.de/api/interpreter'
overpass_query = '''
[out:json][timeout:60];
area["name:en"="Nalgonda"]["admin_level"="5"]->.searchArea;
(
  way["landuse"="farmland"](area.searchArea);
);
out geom 500;
'''

print('Querying Overpass API...')
try:
    data = urllib.parse.urlencode({'data': overpass_query}).encode('utf-8')
    req = urllib.request.Request(overpass_url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        
        features = []
        for element in result.get('elements', []):
            if element['type'] == 'way':
                coords = [[node['lon'], node['lat']] for node in element.get('geometry', [])]
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                if len(coords) >= 4:
                    geom = {"type": "Polygon", "coordinates": [coords]}
                    features.append({"type": "Feature", "geometry": geom, "properties": {"crop": "Farmland"}})
        
        print(f"Found {len(features)} features.")
        if features:
            geojson = {"type": "FeatureCollection", "features": features}
            os.makedirs('data/nalgonda/crop_layers', exist_ok=True)
            with open('data/nalgonda/crop_layers/nalgonda_farmland_osm.geojson', 'w') as f:
                json.dump(geojson, f)
except Exception as e:
    print('Error:', e)
