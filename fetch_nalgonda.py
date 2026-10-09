import urllib.request
import json
import os

url = 'https://nominatim.openstreetmap.org/search.php?q=Nalgonda,+Telangana,+India&polygon_geojson=1&format=jsonv2'
req = urllib.request.Request(url, headers={'User-Agent': 'AcademicGISProject/1.0 (contact@example.com)'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        if data and len(data) > 0:
            for item in data:
                if item.get('osm_type') == 'relation':
                    geojson = item.get('geojson')
                    if geojson:
                        os.makedirs('data', exist_ok=True)
                        with open('data/nalgonda.geojson', 'w') as f:
                            json.dump(geojson, f)
                        print('Successfully downloaded Nalgonda boundary.')
                        exit(0)
            print('No relation geojson found.')
        else:
            print('No data returned.')
except Exception as e:
    print(f'Failed: {e}')
