import requests
import json

overpass_url = 'https://overpass-api.de/api/interpreter'
overpass_query = '''
[out:json][timeout:60];
area["name"="Nalgonda"]->.searchArea;
(
  way["landuse"="farmland"](area.searchArea);
);
out geom 1000;
'''
headers = {'User-Agent': 'CropAcreageBot/1.0'}
response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers)
result = response.json()
print("Elements found:", len(result.get('elements', [])))
