# Crop Acreage and Height Estimation System

## PROJECT OVERVIEW
A full-stack, scientifically rigorous agricultural analysis platform combining Geospatial Information Systems (GIS), Satellite Remote Sensing (Google Earth Engine), and Deep Learning (PyTorch) to provide accurate estimates of crop acreage and physical plant height. 

## ARCHITECTURE
- **Frontend**: Streamlit + Folium (Interactive Mapping)
- **Backend**: FastAPI (Python)
- **Database**: SQLite (Development) / PostgreSQL (Production)
- **Geospatial & Satellite**: Google Earth Engine (GEE) Python API, Sentinel-2 Provider, GeoJSON, Shapely
- **Machine Learning**: PyTorch (ResNet18 Backbone)

## LOCAL RUN
1. Install dependencies: `pip install -r requirements.txt`
2. Run backend: `uvicorn backend.app.main:app --port 8000`
3. Run frontend: `streamlit run frontend/streamlit_app.py --server.port 8501`

## ONLINE DEMO
The application can be deployed using standard Docker Compose or cloud providers.

## ENVIRONMENT VARIABLES
See `.env.example`. Required keys:
- `SECRET_KEY`
- `DATABASE_URL`
- `SATELLITE_PROVIDER` (Set to `gee` or `mock`)
- `API_BASE_URL` (For frontend to talk to backend)

## DEPLOYMENT
We provide a `Dockerfile`, `Dockerfile.frontend`, and `docker-compose.yml` for simplified deployment on platforms like Render, AWS, or Heroku. 
1. Build & Run: `docker-compose up --build -d`

## DEMO MODE
When `SATELLITE_PROVIDER=mock` is set (or GEE credentials are missing), the system falls back to a deterministic Mock Provider. Output is explicitly labelled **DEMO DATA** in the UI to prevent fabrication claims. 

## REAL GEE MODE
When `SATELLITE_PROVIDER=gee` is set, and valid `GEE_SERVICE_ACCOUNT` + `GEE_PRIVATE_KEY_PATH` are provided, the system queries live Sentinel-2 metadata for real-time vegetative bounds.

## MODEL INFORMATION
- **Model**: `rgb_height_v1`
- **Training Dataset**: FIP 1.0 (Winter Wheat, Switzerland)
- **Held-out Test MAE**: 0.112 m
- **Held-out Test R²**: 0.621

## SCIENTIFIC VALIDATION STATUS
- **Indian Validation**: NOT COMPLETED (0 real Indian measured image-height pairs available).
- **Yield Estimation**: The system does NOT calculate yield (Height * Area is scientifically invalid without verified crop-specific biomass allometry equations).

## KNOWN LIMITATIONS
- Height model is highly tailored to the domain of Swiss winter wheat (FIP 1.0) and requires localized fine-tuning for specific Indian crop variants.
