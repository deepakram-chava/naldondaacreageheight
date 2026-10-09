import ee
try:
    ee.Initialize()
    print("GEE initialized successfully!")
except Exception as e:
    print(f"GEE Initialization failed: {e}")
