import numpy as np
import openeo
#import rasterio
#from rasterio.io import MemoryFile
from indexesClass import IndexesCalculations, PlotHeatMap, TimeSeriesCalculator
import datetime 
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

''''NOTE: This is to be updated when the first demo is tested!

def get_data_from_openeo(start, end, dphi, dl, phi, l):
    datacube = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent={"west": l-dl/2, "south": phi-dphi/2, "east": l+dl/2, "north": phi+dphi/2},
    temporal_extent=[start, end],
    bands=["B02", "B03", "B04","B08","sunAzimuthAngles","sunZenithAngles","viewAzimuthMean","viewZenithMean"])
    job = datacube.create_job(out_format="GTiff")
    job.start_and_wait()
    job.get_results().download_files("output_directory") #change that to the name 

    result = datacube.download()

    with MemoryFile(result) as memfile:
        with memfile.open() as src:
            band_data = src.read(1)
            transform = src.transform
    return band_data, transform
    '''


#NOTE: THIS METHOD WILL TEST THE DOWNLINKING OF DATA FROM COPERNICUS HUB
'''
def main_Alexandra():
    start_date = datetime.date(input())
    end_date = datetime.date(input())
    red_band, _ = get_data_from_openeo(start_date, end_date)
    nir_band, _ = get_data_from_openeo("B08")
    indexes = IndexesCalculations(None)
    '''

def create_true_color_image(red, green, blue,gamma=0.5):
        true_color = np.power(np.dstack((red, green, blue)),gamma)
        return true_color

def final_demo():
    t0=time.time()
    bands = IndexesCalculations("celle_cereals/raw_bands/openEO_2022-06-15Z.tif")
    lai, cab = bands.get_lai_and_cab()
    ndvi = bands.calculate_ndvi()
    msavi = bands.calculate_msavi()
    ndwi = bands.calculate_ndwi()

    ndvi = np.array(ndvi, dtype=np.float32)
    msavi = np.array(msavi, dtype=np.float32)
    ndwi = np.array(ndwi, dtype=np.float32)

    red = bands.red_raw()
    blue = bands.blue_raw()
    green = bands.green_raw()

    true_color_Image = create_true_color_image(red, green, blue)

    PlotHeatMap.plot_combined_heatmaps(msavi, ndwi, lai, ndvi, cab, true_color_Image)
    PlotHeatMap.plot_cab_heatmap(cab)
    PlotHeatMap.plot_lai_heatmap(lai)
    PlotHeatMap.plot_msavi_heatmap(msavi)
    PlotHeatMap.plot_ndvi_heatmap(ndvi)
    PlotHeatMap.plot_ndwi_heatmap(ndwi)
    PlotHeatMap.plot_true_color_image(true_color_Image)

    directory="celle_cereals/raw_bands/"
    lai_arrs,cab_arrs,date_strings,t,diffs_lai,diffs_cab=TimeSeriesCalculator(directory).get_time_series()
    PlotHeatMap.plot_lai_heatmap_animations(lai_arrs,date_strings)
    PlotHeatMap.plot_mean_lai_diff_heatmap(diffs_lai)
    PlotHeatMap.plot_mean_cab_diff_heatmap(diffs_cab)
    PlotHeatMap.plot_lai_ill_mask(diffs_lai,true_color_Image)
    PlotHeatMap.plot_cab_ill_mask(diffs_lai,true_color_Image)
    PlotHeatMap.plot_combined_ill_mask(diffs_lai,diffs_cab,true_color_Image)
    print(f"Running time: {(time.time()-t0)/60} min")

def image_demo():
    bands = IndexesCalculations("celle_cereals/raw_bands/openEO_2022-06-15Z.tif")

    red = bands.red_raw()
    blue = bands.blue_raw()
    green = bands.green_raw()

    true_color_Image = create_true_color_image(red, green, blue)
    PlotHeatMap.plot_true_color_image(true_color_Image)
    

try:
    final_demo()
except Exception as e:
    print(f"An error occurred: {e}")
