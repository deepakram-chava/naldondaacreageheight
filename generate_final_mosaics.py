import sys
sys.stdout.reconfigure(line_buffering=True)
import os

# GDAL environment optimizations for robust HTTP / Azure COG streaming
os.environ["GDAL_HTTP_TIMEOUT"] = "25"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF,.vrt"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"

import odc.stac
import planetary_computer
import pystac_client
from odc.geo.geobox import GeoBox
import json
import rasterio.features
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
import time

def build_all_mosaics():
    t0 = time.time()
    cache_dir = 'backend/cache/nalgonda_maps'
    os.makedirs(cache_dir, exist_ok=True)

    with open('data/nalgonda/boundaries/nalgonda.geojson') as f:
        d = json.load(f)
    geom_4326 = d if 'coordinates' in d else d['features'][0]['geometry']
    bbox_4326 = [78.6152855, 16.3556998, 79.7020031, 17.360205]

    print("1. Searching Sentinel-2 scenes on 2024-10-29 pass...")
    cat = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace
    )
    s = cat.search(
        collections=['sentinel-2-l2a'],
        bbox=bbox_4326,
        datetime='2024-10-28/2024-10-30',
        query={'eo:cloud_cover': {'lt': 15}}
    )
    items = [it for it in s.items() if 'T44' in it.id]
    print(f"   Selected {len(items)} UTM Zone 44N scenes covering >95% of Nalgonda:")
    for it in items:
        print(f"   • {it.id} (cloud: {it.properties.get('eo:cloud_cover', 0):.2f}%)")

    # Define target GeoBox for Nalgonda district
    gbox = GeoBox.from_bbox(bbox_4326, crs='EPSG:4326', resolution=0.0025)
    width, height = gbox.shape.x, gbox.shape.y
    print(f"   Target district grid: {width}x{height} pixels (EPSG:4326)")

    # Load required bands one by one, saving to disk cache immediately
    band_names = ['B04', 'B03', 'B02', 'B08', 'B11']
    reprojected_bands = {}

    print("2. Downloading & caching bands across all 4 scenes...")
    for b in band_names:
        npy_path = os.path.join(cache_dir, f'mosaic_band_{b}_2024_Kharif.npy')
        if os.path.exists(npy_path):
            print(f"   ✓ Loaded {b} from disk cache ({npy_path})")
            reprojected_bands[b] = np.load(npy_path)
            continue

        success = False
        for attempt in range(1, 4):
            tb = time.time()
            print(f"   -> Loading {b} across 4 scenes (attempt {attempt}/3, res=300m, EPSG:32644)...", end=" ", flush=True)
            try:
                ds_b = odc.stac.load(
                    items,
                    bands=[b],
                    resolution=300,
                    crs='EPSG:32644',
                    patch_url=planetary_computer.sign
                )
                ds_reproj = ds_b.odc.reproject(gbox).squeeze('time', drop=True)
                band_data = ds_reproj[b].values.astype(float)
                np.save(npy_path, band_data)
                reprojected_bands[b] = band_data
                print(f"done in {time.time() - tb:.1f}s (cached)")
                success = True
                break
            except Exception as e:
                print(f"FAILED ({e})")
                time.sleep(2)
        
        if not success:
            raise RuntimeError(f"Failed to load band {b} after 3 attempts.")

    poly_mask = rasterio.features.geometry_mask(
        [geom_4326],
        out_shape=(height, width),
        transform=gbox.transform,
        invert=True
    )

    b02 = reprojected_bands['B02']
    b03 = reprojected_bands['B03']
    b04 = reprojected_bands['B04']
    b08 = reprojected_bands['B08']
    b11 = reprojected_bands['B11']

    def save_rgb(r, g, b, layer_name):
        rgb = np.dstack((r, g, b)) / 2800.0
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        valid_data = np.any(rgb > 0, axis=-1)
        # Transparent outside district polygon or where NoData
        alpha = np.where(poly_mask & valid_data, 255, 0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))
        img = Image.fromarray(rgba, 'RGBA')
        path = os.path.join(cache_dir, f'sentinel2_{layer_name}_2024_Kharif.png')
        img.save(path)
        vis = np.sum(alpha > 0)
        tot = alpha.size
        print(f"   ✓ Saved {layer_name:12s} continuous mosaic: shape={rgba.shape}, visible={vis:,} ({vis/tot:.1%})")

    def save_index(index_array, layer_name, vmin, vmax, colors):
        index_array = np.nan_to_num(index_array, nan=-1.0)
        index_array = np.clip(index_array, -1.0, 1.0)
        cmap = mcolors.LinearSegmentedColormap.from_list(layer_name, colors)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(index_array))
        rgba[~poly_mask, 3] = 0.0
        rgba[index_array <= -0.99, 3] = 0.0
        rgba = (rgba * 255).astype(np.uint8)
        img = Image.fromarray(rgba, 'RGBA')
        path = os.path.join(cache_dir, f'sentinel2_{layer_name}_2024_Kharif.png')
        img.save(path)
        vis = np.sum(rgba[:, :, 3] > 0)
        tot = rgba[:, :, 3].size
        print(f"   ✓ Saved {layer_name:12s} continuous mosaic: shape={rgba.shape}, visible={vis:,} ({vis/tot:.1%})")

    print("\n3. Generating and clipping all 7 continuous real mosaics...")
    # 1. True Color
    save_rgb(b04, b03, b02, 'true-color')
    # 2. False Color
    save_rgb(b08, b04, b03, 'false-color')
    # 3. NDVI
    ndvi = (b08 - b04) / (b08 + b04 + 1e-8)
    save_index(ndvi, 'ndvi', 0.0, 0.8, ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529'])
    # 4. NDWI
    ndwi = (b03 - b08) / (b03 + b08 + 1e-8)
    save_index(ndwi, 'ndwi', -0.2, 0.3, ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'])
    # 5. NDMI
    ndmi = (b08 - b11) / (b08 + b11 + 1e-8)
    save_index(ndmi, 'ndmi', -0.2, 0.5, ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'])
    # 6. EVI
    evi = 2.5 * ((b08 - b04) / (b08 + 6.0*b04 - 7.5*b02 + 1.0))
    save_index(evi, 'evi', 0.0, 0.8, ['#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837'])
    # 7. SAVI
    savi = ((b08 - b04) / (b08 + b04 + 0.5)) * 1.5
    save_index(savi, 'savi', 0.0, 0.8, ['#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837'])

    print(f"\nALL 7 DISTRICT-CLIPPED MOSAICS COMPLETED IN {time.time() - t0:.1f}s!")

if __name__ == '__main__':
    build_all_mosaics()
