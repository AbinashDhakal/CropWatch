import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import tensorflow as tf

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations if needed
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

class ImageSplicer:
    """Class to handle image splicing."""
    def __init__(self, file_path, n):
        """
        Initializes the ImageSplicer with the file path and number of tiles.
        
        Parameters:
        - file_path: str, path to the image file
        - n: int, number of tiles along one dimension
        """
        self.file_path = file_path
        self.n = n

    def splice_image(self):
        """
        Splices the image into n x n tiles.
        
        Returns:
        - cropped_images: list of PIL images, the spliced image tiles
        """
        img = Image.open(self.file_path)  # Open the image file
        width, height = img.size  # Get the dimensions of the image
        tile_width = width // self.n  # Calculate width of each tile
        tile_height = height // self.n  # Calculate height of each tile
        cropped_images = []

        for i in range(self.n):
            for j in range(self.n):
                left = j * tile_width
                upper = i * tile_height
                right = left + tile_width
                lower = upper + tile_height
                cropped_img = img.crop((left, upper, right, lower))  # Crop the image to create a tile
                cropped_images.append(cropped_img)  # Append the tile to the list
        return cropped_images


class CloudMasker:
    """Class to handle cloud masking and inpainting."""
    def __init__(self, t1=0.1, t2=0.1, kernel_size=10, inpaint_radius=5):
        """
        Initializes the CloudMasker with the given parameters.
        
        Parameters:
        - t1: float, threshold for red-green ratio
        - t2: float, threshold for green-blue ratio
        - kernel_size: int, size of the kernel for dilation
        - inpaint_radius: int, radius for inpainting
        """
        self.t1 = t1
        self.t2 = t2
        self.kernel_size = kernel_size
        self.inpaint_radius = inpaint_radius

    def mask_clouds(self, image):
        """
        Masks the clouds in an image using specified thresholds and inpaints the masked regions.
        
        Parameters:
        - image: PIL image
        
        Returns:
        - cloud_cover: float, percentage of cloud cover
        - inpainted_image: np.array, inpainted image
        """
        image = np.array(image)  # Convert PIL image to numpy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Extract individual color channels
        R = image[:, :, 2].astype(float)
        G = image[:, :, 1].astype(float)
        B = image[:, :, 0].astype(float)

        # Calculate color ratios
        rg = R / G
        gb = G / B

        # Create cloud mask based on color ratios
        cloud_mask1 = np.logical_and(np.abs(rg - 1) < self.t1, np.abs(gb - 1) < self.t2).astype(np.uint8)
        cloud_mask = cv2.dilate(cloud_mask1, kernel=np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8), iterations=1)

        # Inpaint the masked regions
        R_inpaint = cv2.inpaint(R.astype(np.uint8), cloud_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        G_inpaint = cv2.inpaint(G.astype(np.uint8), cloud_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        B_inpaint = cv2.inpaint(B.astype(np.uint8), cloud_mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        # Recombine the inpainted channels into an image
        inpainted_image = np.stack([B_inpaint, G_inpaint, R_inpaint], axis=2)
        cloud_cover = np.sum(cloud_mask) / cloud_mask.size * 100  # Calculate cloud cover percentage

        return cloud_cover, inpainted_image


class ImageProcessor:
    """Main class to process images."""
    def __init__(self, image_path, model_path, n=3, threshold=20):
        """
        Initializes the ImageProcessor with paths and parameters.
        
        Parameters:
        - image_path: str, path to the image file
        - model_path: str, path to the trained model file
        - n: int, number of tiles along one dimension
        - threshold: float, cloud cover threshold for filtering images
        """
        self.image_path = image_path
        self.model_path = model_path
        self.n = n
        self.threshold = threshold

    def process_images(self):
        """
        Processes the images by splicing, masking, and predicting.
        
        - Splices the image into tiles
        - Masks clouds and filters tiles based on cloud cover
        - Predicts the class of each filtered tile
        """
        splicer = ImageSplicer(self.image_path, self.n)  # Create an ImageSplicer instance
        images = splicer.splice_image()  # Splice the image into tiles
        
        masker = CloudMasker()  # Create a CloudMasker instance
        filtered_images = [image for image in images if masker.mask_clouds(image)[0] < self.threshold]  # Filter images based on cloud cover

        if not os.path.exists('downlink'):  # Create output directory if it doesn't exist
            os.makedirs('downlink')

        model = load_model(self.model_path)  # Load the trained model
        predictions = self.predict_images(model, filtered_images)  # Predict classes for the filtered images

        for i, (image, prediction) in enumerate(zip(filtered_images, predictions)):
            if prediction == 1:  # Save images predicted as class 1
                cv2.imwrite(f'downlink/image_{i}.png', np.array(image))  # Save the image

    def predict_images(self, model, images):
        """
        Predicts classes for the given images using the provided model.
        
        Parameters:
        - model: Keras model, trained model for prediction
        - images: list of PIL images, images to be predicted
        
        Returns:
        - predictions: list of int, predicted classes for the images
        """
        resized_images = [cv2.resize(np.array(image), (64, 64))[:, :, :3] for image in images]  # Resize images to 64x64 and ensure 3 channels
        predictions = [model.predict(np.expand_dims(image, axis=0)) for image in resized_images]  # Predict classes
        return [np.argmax(prediction) for prediction in predictions]  # Return the predicted classes


if __name__ == "__main__":
    image_path = r'test1.png'  # Path to the input image
    model_path = r'resnet50_veg.h5'  # Path to the trained model
    image_processor = ImageProcessor(image_path, model_path, n=3, threshold=20)  # Create an ImageProcessor instance
    image_processor.process_images()  # Process the images
