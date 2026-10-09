import json
with open("data/nalgonda/boundaries/nalgonda.geojson", "r") as f:
    data = json.load(f)
    print(data.keys())
    if "type" in data:
        print(f"Type: {data['type']}")
