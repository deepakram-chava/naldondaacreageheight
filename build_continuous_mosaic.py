import sys
sys.stdout.reconfigure(line_buffering=True)
import odc.stac
import planetary_computer
from backend.app.data_sources.planetary_computer_service import pc_service
import json
import rasterio.features
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image
import os
import matplotlib.colors as mcolors
import time

def build_mosaics():
    t0 = time.time()
    with open('data/nalgonda/boundaries/nalgonda.geojson') as f:
        d = json.load(f)
    geom = d if 'coordinates' in d else d['features'][0]['geometry']
    bbox = [78.6152855, 16.3556998, 79.7020031, 17.360205]

    print('1. Searching Sentinel-2 scenes for 2024-10-29 orbit pass...')
    res = pc_service.search_sentinel2(bbox, '2024-10-28', '2024-10-30', max_cloud_cover=15)
    # The primary UTM Zone 44N tiles that cover >95% of Nalgonda
    items = [it for it in res['items'] if 'T44' in it.id]
    print(f'   Selected {len(items)} UTM Zone 44N scenes covering Nalgonda:')
    for it in items:
        print(f'   • {it.id} (cloud: {it.properties.get("eo:cloud_cover", 0):.2f}%)')

    print('2. Downloading & mosaicking bands via odc.stac (EPSG:4326, res=0.002)...')
    ds = odc.stac.load(
        items,
        bands=['B02', 'B03', 'B04', 'B08', 'B11'],
        bbox=bbox,
        crs='EPSG:4326',
        resolution=0.002,
        patch_url=planetary_computer.sign,
        groupby='solar_day'
    )
    print(f'   ✓ Loaded all bands in {time.time() - t0:.1f}s! Dims: {dict(ds.dims)}')

    ds = ds.squeeze('time', drop=True)
    height, width = ds.B04.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
    poly_mask = rasterio.features.geometry_mask([geom], out_shape=(height, width), transform=transform, invert=True)

    b02 = ds.B02.values.astype(float)
    b03 = ds.B03.values.astype(float)
    b04 = ds.B04.values.astype(float)
    b08 = ds.B08.values.astype(float)
    b11 = ds.B11.values.astype(float)

    cache_dir = 'backend/cache/nalgonda_maps'
    os.makedirs(cache_dir, exist_ok=True)

    def save_rgb(r, g, b, layer_name):
        rgb = np.dstack((r, g, b)) / 2800.0
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        valid_data = np.any(rgb > 0, axis=-1)
        alpha = np.where(poly_mask & valid_data, 255, 0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))
        img = Image.fromarray(rgba, 'RGBA')
        path = os.path.join(cache_dir, f'sentinel2_{layer_name}_2024_Kharif.png')
        img.save(path)
        print(f'   ✓ Saved {layer_name} continuous mosaic: shape={rgba.shape}, visible_pixels={np.sum(alpha > 0):,}')

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
        print(f'   ✓ Saved {layer_name} continuous mosaic: shape={rgba.shape}, visible_pixels={np.sum(rgba[:, :, 3] > 0):,}')

    print('3. Rendering and boundary-clipping all 7 layers...')
    save_rgb(b04, b03, b02, 'true-color')
    save_rgb(b08, b04, b03, 'false-color')

    ndvi = (b08 - b04) / (b08 + b04 + 1e-8)
    save_index(ndvi, 'ndvi', 0.0, 0.8, ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529'])

    ndwi = (b03 - b08) / (b03 + b08 + 1e-8)
    save_index(ndwi, 'ndwi', -0.2, 0.3, ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'])

    ndmi = (b08 - b11) / (b08 + b11 + 1e-8)
    save_index(ndmi, 'ndmi', -0.2, 0.5, ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'])

    evi = 2.5 * ((b08 - b04) / (b08 + 6.0*b04 - 7.5*b02 + 1.0))
    save_index(evi, 'evi', 0.0, 0.8, ['#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837'])

    savi = ((b08 - b04) / (b08 + b04 + 0.5)) * 1.5
    save_index(savi, 'savi', 0.0, 0.8, ['#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837'])

    print(f'ALL 7 DISTRICT-CLIPPED MOSAICS COMPLETED IN {time.time() - t0:.1f}s!')

if __name__ == '__main__':
    build_mosaics()
