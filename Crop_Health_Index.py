# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:38:20 2024

@author: abudh
"""


import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import openeo

def load_data_from_openeo():
    # Connect to the OpenEO back-end
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu")

    # Authenticate with the back-end
    connection.authenticate_oidc()

    # Load Sentinel-2 data collection
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=("2022-05-01", "2022-05-30"),
        spatial_extent={
            "west": 3.20,
            "south": 51.18,
            "east": 3.25,
            "north": 51.21,
            "crs": "EPSG:4326",
        },
        bands=["B08", "B07", "B06", "B05", "B04", "B03", "B02"],
        max_cloud_cover=50,
    )
    
    # Compute the median for the temporal extent
    s2_median = s2_cube.reduce_dimension(dimension="t", reducer="median")
    
    # Download the results as an xarray dataset
    job = s2_median.execute_batch("output.nc", out_format="netCDF")
    job.download_results(".")

    ds = xr.open_dataset("output.nc")
    return ds

def load_data_from_directory(data_folder_path):
    pattern = "*.tiff"
    search_pattern = f"{data_folder_path}\\{pattern}"

    # Use glob to find files matching the pattern
    S_sentinel_bands = glob.glob(search_pattern)
    print("Files found:", S_sentinel_bands)

    if S_sentinel_bands:
        # Initialize a list to store band data and band names
        band_data = []
        band_names = []

        # Read each TIFF file and append the first band to the list
        for file_path in S_sentinel_bands:
            try:
                with rasterio.open(file_path) as ds:
                    band_data.append(ds.read(1))
                    band_names.append(ds.descriptions[0])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Stack all bands into a single array
        arr_st = np.stack(band_data)
        print("Shape of stacked bands:", arr_st.shape)

        # Create an xarray dataset
        ds = xr.Dataset({f"B{str(i+1).zfill(2)}": (["y", "x"], arr_st[i]) for i in range(len(arr_st))})
        return ds
    else:
        print("No files found.")
        return None

# Functions to calculate various indices
def calculate_evi(ds):
    nir = ds["B08"] * 0.0001
    red = ds["B04"] * 0.0001
    blue = ds["B02"] * 0.0001
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    return evi

def calculate_lai(ds):
    evi = calculate_evi(ds)
    lai = (3.618 * evi - 0.118)
    return lai

def calculate_rvi(ds):
    nir = ds["B08"] * 0.0001
    red = ds["B04"] * 0.0001
    rvi = nir / red
    return rvi

def calculate_pssra(ds):
    nir = ds["B07"] * 0.0001
    red = ds["B04"] * 0.0001
    pssra = nir / red
    return pssra

def calculate_ndvi(ds):
    nir = ds["B08"] * 0.0001
    red = ds["B04"] * 0.0001
    ndvi = (nir - red) / (nir + red)
    return ndvi

def calculate_ndi45(ds):
    nir = ds["B05"] * 0.0001
    red = ds["B04"] * 0.0001
    ndi45 = (nir - red) / (nir + red)
    return ndi45

def calculate_gndvi(ds):
    nir = ds["B08"] * 0.0001
    green = ds["B03"] * 0.0001
    gndvi = (nir - green) / (nir + green)
    return gndvi

def calculate_mcari(ds):
    red2 = ds["B05"] * 0.0001
    red1 = ds["B04"] * 0.0001
    green = ds["B03"] * 0.0001
    mcari = ((red2 - red1) - 0.2 * (red2 - green)) * (red2 / red1)
    return mcari

def calculate_s2rep(ds):
    red1 = ds["B04"] * 0.0001
    nir = ds["B07"] * 0.0001
    red2 = ds["B05"] * 0.0001
    red3 = ds["B06"] * 0.0001
    s2rep = 705 + 35 * ((red1 + nir) / 2 - red2) / (red3 - red2)
    return s2rep

def calculate_ireci(ds):
    nir = ds["B07"] * 0.0001
    red1 = ds["B04"] * 0.0001
    red2 = ds["B05"] * 0.0001
    red3 = ds["B06"] * 0.0001
    ireci = (nir - red1) / (red2 / red3)
    return ireci

def calculate_savi(ds, L=0.428):
    nir = ds["B08"] * 0.0001
    red = ds["B04"] * 0.0001
    savi = (1 + L) * (nir - red) / (nir + red + L)
    return savi

def calculate_indices(ds):
    indices = {
        "NDVI": calculate_ndvi(ds),
        "SAVI": calculate_savi(ds),
        "RVI": calculate_rvi(ds),
        "PSSRa": calculate_pssra(ds),
        "NDI45": calculate_ndi45(ds),
        "GNDVI": calculate_gndvi(ds),
        "MCARI": calculate_mcari(ds),
        "S2REP": calculate_s2rep(ds),
        "IRECI": calculate_ireci(ds),
        "LAI": calculate_lai(ds),
        "EVI": calculate_evi(ds),
    }
    return indices

def plot_images(ds, index, title):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=90)

    red = ds["B04"] * 0.0001
    green = ds["B03"] * 0.0001
    blue = ds["B02"] * 0.0001
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")

    cax2 = axes[1].imshow(index, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title(title)
    axes[1].set_xlabel("Pixel X")
    axes[1].set_ylabel("Pixel Y")
    
    cbar2 = plt.colorbar(cax2, ax=axes[1], orientation='vertical')
    cbar2.set_label(title)
    
    plt.tight_layout()
    plt.show()

def select_data_source():
    choice = input("Choose data source (1: OpenEO, 2: Local Directory): ").strip()

    if choice == '1':
        ds = load_data_from_openeo()
        print("Data loaded from OpenEO.")
    elif choice == '2':
        # Provide a hint for the directory path using a raw string
        default_path = r"C:\Users\abudh\Desktop\CropWatch\SunderabanData"
        print(f"Hint: The path to the data folder (e.g., {default_path})")
        
        # Prompt the user for the path, using the default path if none is provided
        data_folder_path = input(f"Enter the path to the data folder [{default_path}]: ").strip()
        if not data_folder_path:
            data_folder_path = default_path
        
        # Load data from the local directory
        ds = load_data_from_directory(data_folder_path)
        if ds is None:
            print("No data found or could not load data from the directory.")
            return
        
        print("Data loaded from local directory.")
    else:
        print("Invalid choice. Please select 1 or 2.")
        return

    # Calculate indices
    indices = calculate_indices(ds)
    
    # Plot results
    for index_name, index_data in indices.items():
        plot_images(ds, index_data, index_name)

