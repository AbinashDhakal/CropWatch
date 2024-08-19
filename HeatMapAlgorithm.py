import numpy as np
import matplotlib.pyplot as plt
import openeo
import rasterio
from rasterio.io import MemoryFile

def load_band_from_openeo(band_name):
    # Connect to OpenEO
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    # Load Sentinel-2 data
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=("2022-05-01", "2022-05-30"),
        spatial_extent={
            "west": 4.00,  # Updated coordinates
            "south": 50.50,  # Updated coordinates
            "east": 4.10,  # Updated coordinates
            "north": 50.60,  # Updated coordinates
            "crs": "EPSG:4326",
        },
        bands=[band_name],
        max_cloud_cover=50,
    )

    # Download the data as a binary object
    result = s2_cube.download()

    # If result is in bytes format, we need to handle it properly
    with MemoryFile(result) as memfile:
        with memfile.open() as src:
            band_data = src.read(1)
            transform = src.transform

    return band_data, transform

def calculate_ndvi(nir, red):
    np.seterr(divide='ignore', invalid='ignore')
    # Avoid division by zero by replacing zeroes with a small number
    nir = np.where(nir == 0, 1e-10, nir)
    red = np.where(red == 0, 1e-10, red)
    NDVI = (nir - red) / (nir + red)
    return NDVI

def plot_ndvi_heatmap(ndvi):
    plt.figure(figsize=(10, 10))
    plt.title('NDVI Heat Map')
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

# Main script execution
try:
    # Load Red and NIR bands from OpenEO
    red_band, _ = load_band_from_openeo("B04")
    nir_band, _ = load_band_from_openeo("B08")

    # Calculate NDVI
    NDVI = calculate_ndvi(nir_band, red_band)

    # Plot NDVI heatmap
    plot_ndvi_heatmap(NDVI)

except Exception as e:
    print(f"An error occurred: {e}")
