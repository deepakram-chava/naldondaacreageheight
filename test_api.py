import requests

API_BASE_URL = "http://localhost:8000/api/v1/advanced-gee"
print("Years:", requests.get(f"{API_BASE_URL}/nalgonda/years").json())
print("Seasons:", requests.get(f"{API_BASE_URL}/nalgonda/seasons").json())
print("Acreage:", requests.get(f"{API_BASE_URL}/nalgonda/acreage?year=2024&season=Kharif").json())
