import json
import logging

logging.basicConfig(level=logging.INFO)
from backend.app.data_sources.planetary_computer_service import pc_service
import os

def run_test():
    geom_path = "data/nalgonda/boundaries/nalgonda.geojson"
    with open(geom_path, "r") as f:
        geometry = json.load(f)
        
    result = pc_service.get_nalgonda_data(geometry, "true-color", year="2024", season="Kharif")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    run_test()
