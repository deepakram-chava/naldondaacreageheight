import requests

url = 'http://tracgis.telangana.gov.in/arcgis/rest/services/Agriculture?f=json'
try:
    res = requests.get(url, timeout=5)
    print(res.json())
except Exception as e:
    print('Error:', e)
