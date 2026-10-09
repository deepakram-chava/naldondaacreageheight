import requests
API_BASE_URL = "http://localhost:8000/api/v1/advanced-gee"
print("Years:", requests.get(f"{API_BASE_URL}/nalgonda/years").json())
print("Change:", requests.get(f"{API_BASE_URL}/nalgonda/change?from_year=2023&to_year=2024&season=Kharif").json())
