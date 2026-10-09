import requests
import json
import os

url = 'https://nominatim.openstreetmap.org/search'
params = {
    'q': 'Nalgonda District, Telangana, India',
    'polygon_geojson': 1,
    'format': 'jsonv2'
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, params=params, headers=headers)
if response.status_code == 200:
    data = response.json()
    if data:
        for item in data:
            if item.get('osm_type') == 'relation':
                geojson = item.get('geojson')
                os.makedirs('data/nalgonda/boundaries', exist_ok=True)
                with open('data/nalgonda/boundaries/nalgonda.geojson', 'w') as f:
                    json.dump(geojson, f)
                print('Downloaded!')
                exit(0)
    print('Not found')
else:
    print('Failed', response.status_code)
