import rasterio
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import re
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter

class BandsSentinel:
    def __init__(self, file_path):
        self.raw = rasterio.open(file_path)
        
    def blue_raw(self):
        return self.raw.read(1) * 0.0001
    
    def green_raw(self):
        return self.raw.read(2) * 0.0001
    
    def red_raw(self):
        return self.raw.read(3) * 0.0001
    
    def nir_raw(self):
        return self.raw.read(4) * 0.0001
    
    def sun_azimuth(self):
        return self.raw.read(5)  
    
    def sun_zenith(self):
        return self.raw.read(6)
    
    def view_azimuth(self):
        return self.raw.read(7)
    
    def view_zenith(self):
        return self.raw.read(8)
    
    def blue(self):
        return self.ds["B02"] * 0.0001
    
    def green(self):
        return self.ds["B03"] * 0.0001
    
    def red(self):
        return self.ds["B04"] * 0.0001
    
    def nir(self):
        return self.ds["B05"] * 0.0001
    
    def red_(self):
        return self.ds["B06"] * 0.0001
    
    def red__(self):
        return self.ds["B07"] * 0.0001
    
    def nir_(self):
        return self.ds["B08"] * 0.0001
     
class IndexesCalculations(BandsSentinel):
    def __init__(self, file_path):
        super().__init__(file_path)

    def calculate_evi(self):
        return 2.5 * ((self.nir - self.red) / (self.nir + 6 * self.red - 7.5 * self.blue + 1))
    
    def calculate_rvi(self):
        return self.nir / self.red

    def calculate_pssra(self):
        return self.nir / self.red
    
    def calculate_ndi45(self):
        return (self.nir - self.red) / (self.nir + self.red)

    def calculate_gndvi(self):
        return (self.nir - self.green) / (self.nir + self.green)

    def calculate_mcari(self):
        return ((self.red_ - self.red) - 0.2 * (self.red_ - self.green)) * (self.red_ / self.red)

    def calculate_s2rep(self):
        return 705 + 35 * ((self.red + self.nir) / 2 - self.red_) / (self.red__ - self.red_)
    
    def calculate_ireci(self):
        return (self.nir - self.red) / (self.red_ / self.red__)

    def calculate_savi(self, L=0.428):
        return (1 + L) * (self.nir - self.red) / (self.nir + self.red + L)

    def calculate_ndvi(self):
        red = self.red_raw()
        nir = self.nir_raw()
        np.seterr(divide='ignore', invalid='ignore')
        nir = np.where(nir == 0, 1e-10, nir)
        red = np.where(red == 0, 1e-10, red)
        NDVI = (nir - red) / (nir + red)
        return NDVI
    
    def get_lai_and_cab(self):
        directory = '8.08_test_model6.2'  # model directory
        model = tf.keras.models.load_model(f'{directory}/final_model/final_model_exported.h5')
        input_scaler = joblib.load(f'{directory}/input_scaler.pkl')
        output_scaler = joblib.load(f'{directory}/output_scaler.pkl')
        B = self.blue_raw()
        G = self.green_raw()
        R = self.red_raw()
        NIR = self.nir_raw()
        sun_azimuth = self.sun_azimuth()
        sun_zenith = self.sun_zenith()
        view_azimuth = self.view_azimuth()
        view_zenith = self.view_zenith()
        data = np.dstack([R, G, B, NIR, sun_zenith, view_zenith, np.abs(view_azimuth - sun_azimuth)])
        h, w, _ = data.shape
        data = data.reshape(-1, 7)
        data_scaled = input_scaler.transform(data)
        dataset = tf.data.Dataset.from_tensor_slices((data_scaled))
        dataset = dataset.batch(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
        y_scaled = model.predict(dataset)
        y = output_scaler.inverse_transform(y_scaled)
        y = y.reshape(h, w, 2)
        return y[:,:,0], y[:,:,1]  # first array is LAI, second is Cab

    def calculate_msavi(self):
        nir = self.nir_raw()
        red = self.red_raw()
        return (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2

    def calculate_ndwi(self):
        green = self.green_raw()
        nir = self.nir_raw()
        return (green - nir) / (green + nir)


class TimeSeriesCalculator:
    def __init__(self,directory):
        self.directory=directory

    def get_time_series(self):
        tif_files = [f for f in os.listdir(self.directory) if f.endswith('.tif')]
        tif_files.sort()
        dates=[]
        date_strings=[]
        lai_arrs=[]
        cab_arrs=[]

        for file in tif_files:
            date_string = re.search(r'\d{4}-\d{2}-\d{2}', file).group()
            date_object = datetime.strptime(date_string, '%Y-%m-%d')
            dates.append(date_object)
            date_strings.append(date_string)
            bands = IndexesCalculations(f"{self.directory}/{file}")
            lai, cab = bands.get_lai_and_cab()
            lai_arrs.append(lai)
            cab_arrs.append(cab)
        t=np.array([(dates[i]-dates[0]).days for i in range(len(dates))])
        diffs_lai=[(lai_arrs[frame]-lai_arrs[frame-1])/(t[frame]-t[frame-1]) for frame in range(1,len(t))]
        diffs_cab=[(cab_arrs[frame]-cab_arrs[frame-1])/(t[frame]-t[frame-1]) for frame in range(1,len(t))]
        return lai_arrs,cab_arrs,date_strings,t,diffs_lai,diffs_cab


class PlotHeatMap:

    @staticmethod
    def plot_true_color_image(true_color_image,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('Reference image')
        plt.imshow(true_color_image)
        plt.axis('off') 
        plt.savefig("OUTPUT/true_color_image.png")
        if plot:
            plt.show()

    @staticmethod
    def plot_combined_heatmaps(msavi, ndwi, lai, ndvi, cab, true_color_image,plot=False):
        plt.figure(figsize=(24, 18))

        # MSAVI subplot
        plt.subplot(2, 3, 1)
        plt.imshow(msavi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='MSAVI')
        plt.gca().xaxis.set_visible(False)  
        plt.gca().yaxis.set_visible(False)  

        #True color image
        plt.subplot(2, 3, 2)
        plt.title('Reference image')
        plt.imshow(true_color_image)
        plt.axis('off') 
        
        # NDWI subplot
        plt.subplot(2, 3, 3)
        plt.imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
        plt.colorbar(label='NDWI')
        plt.gca().xaxis.set_visible(False)  
        plt.gca().yaxis.set_visible(False)  

        # LAI subplot
        plt.subplot(2, 3, 4)
        max_lai = np.max(lai)
        plt.imshow(lai, cmap='jet', vmin=0, vmax=max_lai)
        plt.colorbar(label='LAI')
        plt.gca().xaxis.set_visible(False)  
        plt.gca().yaxis.set_visible(False) 

        # NDVI subplot
        plt.subplot(2, 3, 5)
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        plt.gca().xaxis.set_visible(False)  
        plt.gca().yaxis.set_visible(False) 
       
        # Cab subplot
        plt.subplot(2, 3, 6)
        plt.imshow(cab, cmap='RdYlGn', vmin=0)
        plt.colorbar(label=r'$C_{ab},\,\mathrm{\mu g/cm^3}$')
        plt.gca().xaxis.set_visible(False)  
        plt.gca().yaxis.set_visible(False) 

        plt.tight_layout()
        plt.savefig("OUTPUT/figure_combined_heatmaps.png")
        if plot:
            plt.show()

    @staticmethod
    def plot_msavi_heatmap(msavi,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('MSAVI Heat Map')
        plt.imshow(msavi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='MSAVI')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig("OUTPUT/figure_heatmap_msavi.png")
        if plot:
            plt.show()

    @staticmethod
    def plot_ndwi_heatmap(ndwi,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('NDWI Heat Map')
        plt.imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
        plt.colorbar(label='NDWI')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig("OUTPUT/figure_heatmap_ndwi.png")
        if plot:
            plt.show()

    @staticmethod
    def plot_ndvi_heatmap(ndvi,dpi=800,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('NDVI Heat Map')
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig("OUTPUT/figure_heatmap_ndvi.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_lai_heatmap(lai,dpi=800,plot=False):
        max_lai = np.max(lai)
        plt.figure(figsize=(12, 12))
        plt.imshow(lai, cmap='jet', vmin=0, vmax=max_lai)
        cbar = plt.colorbar(label='Leaf Area Index (LAI)', orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        ax = plt.gca()
        rect = Rectangle((0, 0), lai.shape[1], lai.shape[0], linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tight_layout()
        plt.savefig("OUTPUT/figure_heatmap_lai.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_lai_heatmap_animations(lai_arrs,date_strings,dpi=800):
        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots()
        img=ax.imshow(lai_arrs[0], cmap='jet', vmin=0, vmax=8)
        cbar = fig.colorbar(img,label='Leaf Area Index (LAI)', orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        rect = Rectangle((0, 0), lai_arrs[0].shape[1], lai_arrs[0].shape[0], linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        def _update(frame):
            img.set_data(lai_arrs[frame])
            ax.set_title(date_strings[frame])
            return [img]
        ani = animation.FuncAnimation(fig, _update, frames=len(lai_arrs), interval=2000, blit=True)
        ani.save('OUTPUT/TIME_SERIES/lai_heatmaps_animation.gif',dpi=dpi)

    @staticmethod
    def plot_lai_diffs_heatmap_animations(diffs_lai,date_strings,dpi=800):
        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots()
        img=ax.imshow(diffs_lai[0], cmap='jet')
        cbar = fig.colorbar(img,label=r'LAI rate of change, $day^{-1}$', orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        rect = Rectangle((0, 0), diffs_lai[0].shape[1], diffs_lai[0].shape[0], linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        def _update(frame):
            img.set_data(diffs_lai[frame])
            ax.set_title(date_strings[frame])
            return [img]
        ani = animation.FuncAnimation(fig, _update, frames=len(diffs_lai), interval=2000, blit=True)
        ani.save('OUTPUT/TIME_SERIES/lai_diffs_heatmaps_animation.gif',dpi=dpi)

    @staticmethod
    def plot_mean_lai_diff_heatmap(diffs_lai,dpi=800,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('Leaf area index rate of change')
        plt.imshow(np.mean(np.dstack(diffs_lai),axis=2), cmap='RdYlGn')
        plt.colorbar(label=r'LAI rate of change, $day^{-1}$')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.legend()
        plt.savefig("OUTPUT/TIME_SERIES/figure_heatmap_mean_lai_diff.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_cab_heatmap(cab,dpi=800,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('Chlorophyl a+b concentration Heat Map')
        plt.imshow(cab, cmap='RdYlGn', vmin=0)
        plt.colorbar(label=r'$C_{ab},\,\mathrm{\mu g/cm^3}$')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.legend()
        plt.savefig("OUTPUT/figure_heatmap_cab.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_cab_heatmap_animations(cab_arrs,date_strings,dpi=800):
        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots()
        img=ax.imshow(cab_arrs[0], cmap='jet', vmin=0)
        cbar = fig.colorbar(img,label=r'$C_{ab},\,\mathrm{\mu g/cm^3}$', orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        rect = Rectangle((0, 0), cab_arrs[0].shape[1], cab_arrs[0].shape[0], linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        def _update(frame):
            img.set_data(cab_arrs[frame])
            ax.set_title(date_strings[frame])
            return [img]
        ani = animation.FuncAnimation(fig, _update, frames=len(cab_arrs), interval=2000, blit=True)
        ani.save('OUTPUT/TIME_SERIES/cab_heatmaps_animation.gif',dpi=dpi)

    @staticmethod
    def plot_cab_diffs_heatmap_animations(diffs_cab,date_strings,dpi=800):
        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots()
        img=ax.imshow(diffs_cab[0], cmap='jet')
        cbar = fig.colorbar(img,label=r'$C_{ab}$ rate of change, $\frac{\mathrm{\mu g/cm^3}}{\mathrm{day}}$', orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        rect = Rectangle((0, 0), diffs_cab[0].shape[1], diffs_cab[0].shape[0], linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        def _update(frame):
            img.set_data(diffs_cab[frame])
            ax.set_title(date_strings[frame])
            return [img]
        ani = animation.FuncAnimation(fig, _update, frames=len(diffs_cab), interval=2000, blit=True)
        ani.save('OUTPUT/TIME_SERIES/cab_diffs_heatmaps_animation.gif',dpi=dpi)

    @staticmethod
    def plot_mean_cab_diff_heatmap(diffs_cab,dpi=800,plot=False):
        plt.figure(figsize=(10, 10))
        plt.title('Chlorophyl a+b concentration rate of change')
        plt.imshow(np.mean(np.dstack(diffs_cab),axis=2), cmap='RdYlGn')
        plt.colorbar(label=r'$C_{ab}$ rate of change, $\frac{\mathrm{\mu g/cm^3}}{\mathrm{day}}$')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.legend()
        plt.savefig("OUTPUT/TIME_SERIES/figure_heatmap_mean_cab_diff.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_lai_ill_mask(diffs_lai,image,dpi=800,plot=False):
        mean_diffs_lai=np.mean(np.dstack(diffs_lai),axis=2)
        #blurred_diffs_lai=cv2.GaussianBlur(mean_diffs_lai,(11,11),0)
        blurred_diffs_lai=gaussian_filter(mean_diffs_lai,2,radius=5)
        mask_lai=(blurred_diffs_lai<0).astype(np.float32)
        mask_lai2=np.dstack([mask_lai,np.zeros_like(mask_lai),np.zeros_like(mask_lai)])
        plt.figure(figsize=(10,10))
        plt.figure(figsize=(10,10))
        plt.title("LAI-based mask")
        cutted=image*(1-mask_lai2)*0.8
        res=np.nan_to_num(cutted+mask_lai2)
        plt.imshow(res)
        plt.savefig("OUTPUT/TIME_SERIES/lai_mask.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_cab_ill_mask(diffs_cab,image,dpi=800,plot=False):
        mean_diffs_cab=np.mean(np.dstack(diffs_cab),axis=2)
        #blurred_diffs_cab=cv2.GaussianBlur(mean_diffs_cab,(11,11),0)
        blurred_diffs_cab=gaussian_filter(mean_diffs_cab,2,radius=5)
        mask_cab=(blurred_diffs_cab<0).astype(np.float32)
        mask_cab2=np.dstack([mask_cab,np.zeros_like(mask_cab),np.zeros_like(mask_cab)])
        plt.figure(figsize=(10,10))
        plt.figure(figsize=(10,10))
        plt.title(r"$C_{ab}$-based mask")
        cutted=image*(1-mask_cab2)*0.8
        res=np.nan_to_num(cutted+mask_cab2)
        plt.imshow(res)
        plt.savefig("OUTPUT/TIME_SERIES/cab_mask.png",dpi=dpi)
        if plot:
            plt.show()

    @staticmethod
    def plot_combined_ill_mask(diffs_lai,diffs_cab,image,dpi=800,plot=False):
        mean_diffs_cab=np.mean(np.dstack(diffs_cab),axis=2)
        mean_diffs_lai=np.mean(np.dstack(diffs_lai),axis=2)
        blurred_diffs_cab=gaussian_filter(mean_diffs_cab,2,radius=5)
        blurred_diffs_lai=gaussian_filter(mean_diffs_lai,2,radius=5)
        mask=np.logical_or(blurred_diffs_cab<0,blurred_diffs_lai<0).astype(np.uint8)
        mask2=np.dstack([mask,np.zeros_like(mask),np.zeros_like(mask)])
        plt.figure(figsize=(10,10))
        plt.title(r"Combined mask")
        cutted=image*(1-mask2)*0.8
        res=np.nan_to_num(cutted+mask2)
        plt.imshow(res)
        plt.savefig("OUTPUT/TIME_SERIES/combined_mask.png",dpi=dpi)
        if plot:
            plt.show()