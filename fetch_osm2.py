import requests

overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
[out:json][timeout:25];
area["name:en"="Nalgonda"]["admin_level"="5"]->.searchArea;
(
  way["landuse"="farmland"](area.searchArea);
);
out geom 100;
"""
print("Querying Overpass API...")
response = requests.post(overpass_url, data={'data': overpass_query})
print(response.status_code)
print(response.text[:200])
